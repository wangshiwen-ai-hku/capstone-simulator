import { expect, test } from '@playwright/test';

test('loads the production UI and completes a workflow through the live API', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  const healthResponse = page.waitForResponse((response) => (
    response.request().method() === 'GET'
    && new URL(response.url()).pathname === '/api/health'
  ));
  const sceneResponse = page.waitForResponse((response) => (
    response.request().method() === 'POST'
    && new URL(response.url()).pathname === '/api/generate-scene'
  ));

  await page.goto('/');

  await expect(page.getByText('MARS Studio')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Run' })).toBeEnabled();
  await expect(page.getByText('API offline')).toHaveCount(0);
  expect((await healthResponse).status()).toBe(200);
  expect((await sceneResponse).status()).toBe(200);

  const easySceneResponse = page.waitForResponse((response) => (
    response.request().method() === 'POST'
    && new URL(response.url()).pathname === '/api/generate-scene'
  ));
  const easyDifficulty = page.getByRole('button', { name: 'easy' });
  await easyDifficulty.click();
  await expect(easyDifficulty).toHaveClass(/active/);
  await page.getByRole('button', { name: 'Apply settings' }).click();
  expect((await easySceneResponse).status()).toBe(200);
  await expect(page.getByRole('button', { name: 'Run' })).toBeEnabled();

  const acceptedResponse = page.waitForResponse((response) => (
    response.request().method() === 'POST'
    && new URL(response.url()).pathname === '/api/runtime/workflows'
  ));
  const completedResponse = page.waitForResponse(
    async (response) => {
      if (
        response.request().method() !== 'GET'
        || !/^\/api\/runtime\/workflows\/[^/]+$/.test(new URL(response.url()).pathname)
        || response.status() !== 200
      ) {
        return false;
      }
      const payload = await response.json();
      return payload.status === 'succeeded' && Boolean(payload.result);
    },
    { timeout: 75_000 },
  );

  await page.getByRole('button', { name: 'Run' }).click();

  expect((await acceptedResponse).status()).toBe(202);
  await completedResponse;
  const runtimeMetrics = page.getByLabel('Runtime metrics');
  await expect(runtimeMetrics).toBeVisible({ timeout: 60_000 });
  await expect(runtimeMetrics.getByText('succeeded')).toBeVisible();
  await expect(page.locator('.runtime-state')).toHaveText(/running|complete/);
  await expect(page.getByRole('alert')).toHaveCount(0);
  expect(pageErrors).toEqual([]);
});
