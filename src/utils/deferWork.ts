/** Yield to the browser so loading UI can paint before heavy synchronous work. */
export function deferWork(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, 0)
  })
}

export async function runDeferred<T>(fn: () => T): Promise<T> {
  await deferWork()
  return fn()
}
