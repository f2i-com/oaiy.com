/**
 * Driving the Providers page the way a person does: the tests never write to the providers origin's database themselves, so a
 * provider in a test was added through the form, with its key typed into the password field of the top-level page.
 */

/**
 * Add a provider through the Providers page (a top-level page of the providers origin) and wait until it is listed.
 *
 * @param {import('playwright').Page} page a page that is NOT in a frame
 * @param {string} providersOrigin
 * @param {{ preset?: string, serverKind?: string, name: string, baseUrl?: string, key?: string, model?: string, check?: boolean }} details
 *   `preset` is the label on its tile; `check` also runs "Check connection" and expects it to work before saving.
 */
export async function addProvider(page, providersOrigin, details) {
  if (!page.url().startsWith(providersOrigin)) await page.goto(`${providersOrigin}/`);
  await page.locator('button.tile', { hasText: details.preset ?? 'A server on this computer' }).click();
  if (details.serverKind) await page.locator('select[aria-label="Which server"]').selectOption(details.serverKind);
  const form = page.locator('section[aria-label="Add or change a provider"]');
  await form.locator('input[type=text]').first().fill(details.name);
  if (details.baseUrl !== undefined) await form.locator('input[type=url]').fill(details.baseUrl);
  if (details.key !== undefined) await form.locator('input[type=password]').fill(details.key);
  if (details.model !== undefined) await form.locator('input[placeholder="or type a model name"]').fill(details.model);
  if (details.check) {
    await form.getByRole('button', { name: 'Check connection' }).click();
    await form.locator('.result.good').waitFor({ timeout: details.timeout ?? 10000 });
  }
  await form.getByRole('button', { name: 'Add provider' }).click();
  await page.locator('section[aria-label="Your providers"] li.row', { hasText: details.name }).waitFor({ timeout: details.timeout ?? 10000 });
}

/** The names listed on the Providers page. */
export async function listedNames(page) {
  return page.locator('section[aria-label="Your providers"] li.row strong').allTextContents();
}

/** Open a provider's form to change it, on the Providers page. */
export async function editProvider(page, name) {
  await page.locator('section[aria-label="Your providers"] li.row', { hasText: name }).getByRole('button', { name: 'Edit' }).click();
  return page.locator('section[aria-label="Add or change a provider"]');
}
