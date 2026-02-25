import { test, expect } from './test-utils';

test.describe('Dashboard Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load successfully', async ({ page }) => {
    await expect(page).toHaveTitle(/GAAP/i);
    await expect(page.getByRole('heading', { name: /dashboard/i })).toBeVisible();
  });

  test('should display sidebar navigation', async ({ page }) => {
    await expect(page.getByRole('navigation')).toBeVisible();
    await expect(page.getByText('GAAP')).toBeVisible();
  });

  test('should display system status card', async ({ page }) => {
    await expect(page.getByText('System Status').or(page.getByText('System Online'))).toBeVisible();
  });

  test('should display budget gauge', async ({ page }) => {
    await expect(page.getByText('Budget').or(page.getByText('Budget Gauge'))).toBeVisible();
  });

  test('should display provider health', async ({ page }) => {
    await expect(page.getByText('Provider').or(page.getByText('Providers'))).toBeVisible();
  });

  test('should navigate to providers page', async ({ page }) => {
    await page.getByRole('link', { name: /providers/i }).click();
    await expect(page).toHaveURL(/.*\/providers/);
    await expect(page.getByRole('heading', { name: /providers/i })).toBeVisible();
  });

  test('should navigate to sessions page', async ({ page }) => {
    await page.getByRole('link', { name: /sessions/i }).click();
    await expect(page).toHaveURL(/.*\/sessions/);
    await expect(page.getByRole('heading', { name: /sessions/i })).toBeVisible();
  });

  test('should navigate to config page', async ({ page }) => {
    await page.getByRole('link', { name: /config/i }).click();
    await expect(page).toHaveURL(/.*\/config/);
  });

  test('should navigate to research page', async ({ page }) => {
    await page.getByRole('link', { name: /research/i }).click();
    await expect(page).toHaveURL(/.*\/research/);
  });

  test('should navigate to healing page', async ({ page }) => {
    await page.getByRole('link', { name: /healing/i }).click();
    await expect(page).toHaveURL(/.*\/healing/);
  });

  test('should navigate to memory page', async ({ page }) => {
    await page.getByRole('link', { name: /memory/i }).click();
    await expect(page).toHaveURL(/.*\/memory/);
  });

  test('should navigate to dream page', async ({ page }) => {
    await page.getByRole('link', { name: /dream/i }).click();
    await expect(page).toHaveURL(/.*\/dream/);
  });

  test('should navigate to budget page', async ({ page }) => {
    await page.getByRole('link', { name: /budget/i }).click();
    await expect(page).toHaveURL(/.*\/budget/);
  });

  test('should navigate to security page', async ({ page }) => {
    await page.getByRole('link', { name: /security/i }).click();
    await expect(page).toHaveURL(/.*\/security/);
  });

  test('should navigate to debt page', async ({ page }) => {
    await page.getByRole('link', { name: /debt/i }).click();
    await expect(page).toHaveURL(/.*\/debt/);
  });

  test('should display layer navigation', async ({ page }) => {
    await expect(page.getByText(/strategy|tactics|execution/i)).toBeVisible();
  });
});
