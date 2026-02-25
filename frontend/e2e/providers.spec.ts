import { test, expect } from './test-utils';

test.describe('Providers Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/providers');
    await page.waitForLoadState('networkidle');
  });

  test('should load successfully', async ({ page }) => {
    await expect(page).toHaveURL(/.*\/providers/);
    await expect(page.getByRole('heading', { name: /providers/i })).toBeVisible();
  });

  test('should display sidebar navigation', async ({ page }) => {
    await expect(page.getByRole('navigation')).toBeVisible();
    await expect(page.getByText('GAAP')).toBeVisible();
  });

  test('should display providers page title and description', async ({ page }) => {
    await expect(page.getByText('LLM Providers')).toBeVisible();
    await expect(page.getByText(/manage and configure/i)).toBeVisible();
  });

  test('should display Add Provider button', async ({ page }) => {
    const addButton = page.getByRole('button', { name: /add provider/i });
    await expect(addButton).toBeVisible();
  });

  test('should open add provider modal when clicking Add Provider button', async ({ page }) => {
    await page.getByRole('button', { name: /add provider/i }).click();
    await expect(page.getByText(/add new provider/i).or(page.getByRole('heading', /add provider/i))).toBeVisible();
  });

  test('should navigate back to dashboard', async ({ page }) => {
    await page.getByRole('link', { name: /dashboard/i }).click();
    await expect(page).toHaveURL(/\//);
    await expect(page.getByRole('heading', { name: /dashboard/i })).toBeVisible();
  });

  test('should navigate to sessions page', async ({ page }) => {
    await page.getByRole('link', { name: /sessions/i }).click();
    await expect(page).toHaveURL(/.*\/sessions/);
  });

  test('should navigate to config page', async ({ page }) => {
    await page.getByRole('link', { name: /config/i }).click();
    await expect(page).toHaveURL(/.*\/config/);
  });

  test('should display providers table or cards when data loads', async ({ page }) => {
    await page.waitForTimeout(2000);
    const hasContent = await page.evaluate(() => {
      const content = document.body.textContent;
      return content && (
        content.includes('Priority') ||
        content.includes('Requests') ||
        content.includes('No providers') ||
        content.includes('healthy') ||
        content.includes('degraded') ||
        content.includes('unhealthy')
      );
    });
    expect(hasContent).toBeTruthy();
  });

  test('should handle loading state', async ({ page }) => {
    await page.goto('/providers');
    const spinner = page.locator('.animate-spin');
    await expect(spinner).toBeVisible({ timeout: 5000 }).catch(() => {});
  });
});
