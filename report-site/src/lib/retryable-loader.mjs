export function createRetryableLoader(load) {
  if (typeof load !== "function") {
    throw new TypeError("load must be a function");
  }

  let pending = null;
  return Object.freeze({
    load() {
      if (!pending) {
        const attempt = Promise.resolve().then(load);
        const guarded = attempt.catch((reason) => {
          if (pending === guarded) {
            pending = null;
          }
          throw reason;
        });
        pending = guarded;
      }
      return pending;
    },
    reset() {
      pending = null;
    },
  });
}
