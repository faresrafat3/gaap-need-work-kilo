#!/usr/bin/env node
/**
 * DeepSeek PoW Solver - Node.js implementation of DeepSeek's custom Keccak-256
 * 
 * This is NOT standard SHA3-256 or Keccak-256. DeepSeek uses a modified Keccak
 * sponge that runs only 23 rounds (rounds 1-23, skipping round 0) instead of
 * the standard 24 rounds. This was reverse-engineered from their JS worker
 * chunk 38401 (deepseek_pow_worker_38401.js).
 * 
 * Usage: node deepseek_pow_solver.js <algorithm> <challenge> <salt> <difficulty> <expire_at>
 * Output: JSON with {answer: number} or {error: string}
 */

const { Buffer } = require('buffer');

// ============= Custom Keccak Implementation (23 rounds) =============

const RC32 = new Uint32Array([
    0, 1, 0, 32898, 0x80000000, 32906, 0x80000000, 0x80008000,
    0, 32907, 0, 0x80000001, 0x80000000, 0x80008081, 0x80000000, 32777,
    0, 138, 0, 136, 0, 0x80008009, 0, 0x8000000a,
    0, 0x8000808b, 0x80000000, 139, 0x80000000, 32905, 0x80000000, 32771,
    0x80000000, 32770, 0x80000000, 128, 0, 32778, 0x80000000, 0x8000000a,
    0x80000000, 0x80008081, 0x80000000, 32896, 0, 0x80000001, 0x80000000, 0x80008008,
]);

const V = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1];
const W_ROT = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44];

function absorb32(queue, state) {
    for (let r = 0; r < queue.length; r += 8) {
        const n = r >> 2;
        state[n] ^= queue[r + 7] << 24 | queue[r + 6] << 16 | queue[r + 5] << 8 | queue[r + 4];
        state[n + 1] ^= queue[r + 3] << 24 | queue[r + 2] << 16 | queue[r + 1] << 8 | queue[r];
    }
}

function squeeze32(state, buf) {
    for (let r = 0; r < buf.length; r += 8) {
        const n = r >> 2;
        buf[r] = state[n + 1];
        buf[r + 1] = state[n + 1] >>> 8;
        buf[r + 2] = state[n + 1] >>> 16;
        buf[r + 3] = state[n + 1] >>> 24;
        buf[r + 4] = state[n];
        buf[r + 5] = state[n] >>> 8;
        buf[r + 6] = state[n] >>> 16;
        buf[r + 7] = state[n] >>> 24;
    }
}

function keccakF(A) {
    const C = new Int32Array(10);
    for (let ri = 1; ri < 24; ri++) {
        // theta
        for (let t = 0; t < 5; t++) {
            const n2 = 2 * t;
            C[n2] = A[n2] ^ A[n2 + 10] ^ A[n2 + 20] ^ A[n2 + 30] ^ A[n2 + 40];
            C[n2 + 1] = A[n2 + 1] ^ A[n2 + 11] ^ A[n2 + 21] ^ A[n2 + 31] ^ A[n2 + 41];
        }
        for (let t = 0; t < 5; t++) {
            const ci = ((t + 1) % 5) * 2;
            const o = C[ci], f = C[ci + 1];
            const d0 = C[((t + 4) % 5) * 2] ^ ((o << 1) | (f >>> 31));
            const d1 = C[((t + 4) % 5) * 2 + 1] ^ ((f << 1) | (o >>> 31));
            for (let r = 0; r < 25; r += 5) {
                const idx = (r + t) * 2;
                A[idx] ^= d0;
                A[idx + 1] ^= d1;
            }
        }
        // rho+pi
        let w0 = A[2], w1 = A[3];
        for (let ii = 0; ii < 24; ii++) {
            const tIdx = V[ii], aVal = W_ROT[ii];
            const c0 = A[2 * tIdx], c1 = A[2 * tIdx + 1];
            const aMod = aVal & 31, sMod = (32 - aVal) & 31;
            const v0 = (w0 << aMod) | (w1 >>> sMod);
            const v1 = (w1 << aMod) | (w0 >>> sMod);
            if (aVal < 32) { w0 = v0; w1 = v1; }
            else { w0 = v1; w1 = v0; }
            A[2 * tIdx] = w0; A[2 * tIdx + 1] = w1;
            w0 = c0; w1 = c1;
        }
        // chi
        for (let t = 0; t < 25; t += 5) {
            for (let n = 0; n < 5; n++) {
                C[2 * n] = A[(t + n) * 2];
                C[2 * n + 1] = A[(t + n) * 2 + 1];
            }
            for (let n = 0; n < 5; n++) {
                const idx = (t + n) * 2;
                A[idx] ^= ~C[((n + 1) % 5) * 2] & C[((n + 2) % 5) * 2];
                A[idx + 1] ^= ~C[((n + 1) % 5) * 2 + 1] & C[((n + 2) % 5) * 2 + 1];
            }
        }
        // iota
        const n2 = 2 * ri;
        A[0] ^= RC32[n2];
        A[1] ^= RC32[n2 + 1];
    }
}

class DSKeccak {
    constructor() {
        this.state = new Int32Array(50);
        this.queue = Buffer.alloc(136);
        this.qoff = 0;
    }

    update(str) {
        const data = Buffer.from(str, 'utf8');
        for (let i = 0; i < data.length; i++) {
            this.queue[this.qoff] = data[i];
            this.qoff++;
            if (this.qoff >= 136) {
                absorb32(this.queue, this.state);
                keccakF(this.state);
                this.qoff = 0;
            }
        }
        return this;
    }

    digest() {
        const st = new Int32Array(this.state);
        const q = Buffer.alloc(136);
        this.queue.copy(q);
        q.fill(0, this.qoff);
        q[this.qoff] |= 6;
        q[135] |= 0x80;
        absorb32(q, st);
        keccakF(st);
        const buf = Buffer.alloc(32);
        squeeze32(st, buf);
        return buf.toString('hex');
    }

    copy() {
        const k = new DSKeccak();
        k.state = new Int32Array(this.state);
        this.queue.copy(k.queue);
        k.qoff = this.qoff;
        return k;
    }
}

// ============= Solver =============

function solve(algorithm, challenge, salt, difficulty, expireAt) {
    if (algorithm !== 'DeepSeekHashV1') {
        return { error: `Unsupported algorithm: ${algorithm}` };
    }

    const prefix = `${salt}_${expireAt}_`;
    const base = new DSKeccak();
    base.update(prefix);

    for (let i = 0; i < difficulty; i++) {
        const h = base.copy().update(String(i)).digest();
        if (h === challenge) {
            return { answer: i };
        }
    }

    return { error: `No solution found in ${difficulty} iterations` };
}

// ============= CLI =============

if (process.argv.length >= 7) {
    const [, , algorithm, challenge, salt, difficulty, expireAt] = process.argv;
    const result = solve(algorithm, challenge, salt, parseInt(difficulty), expireAt);
    console.log(JSON.stringify(result));
} else if (process.argv[2] === '--test') {
    // Verify hash matches
    const h = new DSKeccak().update('abc').digest();
    const expected = 'f841106c601ce9be9bc38525e90d4178d47f21dd8eb9f238fc55ffaa4ca94506';
    console.log(JSON.stringify({ test: h === expected, hash: h, expected }));
} else if (process.argv[2] === '--bench') {
    const prefix = 'testprefix_1234567890_';
    const base = new DSKeccak();
    base.update(prefix);
    const start = Date.now();
    const count = 100000;
    for (let i = 0; i < count; i++) {
        base.copy().update(String(i)).digest();
    }
    const elapsed = (Date.now() - start) / 1000;
    console.log(JSON.stringify({ rate: Math.round(count / elapsed), time_144k: (144000 / (count / elapsed)).toFixed(2) + 's' }));
} else {
    console.error('Usage: node deepseek_pow_solver.js <algorithm> <challenge> <salt> <difficulty> <expire_at>');
    console.error('       node deepseek_pow_solver.js --test');
    console.error('       node deepseek_pow_solver.js --bench');
    process.exit(1);
}
