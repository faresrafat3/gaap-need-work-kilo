'use client';

import { useShortcutsHelp } from '@/hooks/useKeyboardShortcuts';
import { Modal, ModalHeader, ModalBody } from './Modal';
import { Badge } from './Badge';

interface KeyboardShortcutsHelpProps {
  isOpen: boolean;
  onClose: () => void;
}

export function KeyboardShortcutsHelp({ isOpen, onClose }: KeyboardShortcutsHelpProps) {
  const { shortcuts } = useShortcutsHelp();

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <ModalHeader>
        <h2 className="text-lg font-semibold">Keyboard Shortcuts</h2>
      </ModalHeader>
      <ModalBody>
        <div className="space-y-2">
          {shortcuts.map((shortcut, index) => (
            <div
              key={index}
              className="flex items-center justify-between py-2 border-b border-layer1/20 last:border-0"
            >
              <span className="text-sm text-gray-400">{shortcut.description}</span>
              <Badge variant="default">{shortcut.keys}</Badge>
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs text-gray-500">
          Press <Badge variant="default">?</Badge> anytime to see this help
        </p>
      </ModalBody>
    </Modal>
  );
}
