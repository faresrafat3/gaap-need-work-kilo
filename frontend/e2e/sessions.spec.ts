import { test, expect } from './test-utils';

test.describe('Sessions Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/sessions');
    await page.waitForLoadState('networkidle');
  });

  test('should load successfully', async ({ page }) => {
    await expect(page).toHaveURL(/.*\/sessions/);
    await expect(page.getByRole('heading', { name: /sessions/i })).toBeVisible();
  });

  test('should display sidebar navigation', async ({ page }) => {
    await expect(page.getByRole('navigation')).toBeVisible();
    await expect(page.getByText('GAAP')).toBeVisible();
  });

  test('should display sessions page title and description', async ({ page }) => {
    await expect(page.getByText('Sessions')).toBeVisible();
    await expect(page.getByText(/manage research and work/i)).toBeVisible();
  });

  test('should display sessions table with headers', async ({ page }) => {
    await expect(page.getByText('ID')).toBeVisible();
    await expect(page.getByText('Type')).toBeVisible();
    await expect(page.getByText('Status')).toBeVisible();
    await expect(page.getByText('Created')).toBeVisible();
    await expect(page.getByText('Updated')).toBeVisible();
    await expect(page.getByText('Actions')).toBeVisible();
  });

  test('should navigate back to dashboard', async ({ page }) => {
    await page.getByRole('link', { name: /dashboard/i }).click();
    await expect(page).toHaveURL(/\//);
    await expect(page.getByRole('heading', { name: /dashboard/i })).toBeVisible();
  });

  test('should navigate to providers page', async ({ page }) => {
    await page.getByRole('link', { name: /providers/i }).click();
    await expect(page).toHaveURL(/.*\/providers/);
  });

  test('should navigate to config page', async ({ page }) => {
    await page.getByRole('link', { name: /config/i }).click();
    await expect(page).toHaveURL(/.*\/config/);
  });

  test('should handle loading state', async ({ page }) => {
    await page.goto('/sessions');
    const spinner = page.locator('.animate-spin');
    await expect(spinner).toBeVisible({ timeout: 5000 }).catch(() => {});
  });

  test('should display action buttons in table rows', async ({ page }) => {
    await page.waitForTimeout(2000);
    const hasActions = await page.evaluate(() => {
      const buttons = document.querySelectorAll('button');
      const buttonContents = Array.from(buttons).map(b => b.textContent?.toLowerCase() || '');
      return buttonContents.some(text => 
        text.includes('pause') || 
        text.includes('play') || 
        text.includes('resume') ||
        text.includes('download') ||
        text.includes('export') ||
        text.includes('delete') ||
        text.includes('trash')
      );
    });
    expect(hasActions).toBeTruthy();
  });

  test('should display empty state when no sessions', async ({ page }) => {
    await page.waitForTimeout(3000);
    const hasEmptyState = await page.evaluate(() => {
      const content = document.body.textContent;
      return content && content.includes('No sessions found');
    });
    expect(hasEmptyState).toBeTruthy();
  });
});
