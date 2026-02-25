import { test as base } from '@playwright/test';

export const test = base.extend({
  page: async ({ page }, use) => {
    const consoleErrors: string[] = [];
    
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    page.on('pageerror', (error) => {
      consoleErrors.push(error.message);
    });

    await use(page);

    if (consoleErrors.length > 0) {
      console.log('Console errors:', consoleErrors);
    }
  },
});

export { expect } from '@playwright/test';
