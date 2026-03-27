import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatPriceLabel(isFree: boolean) {
  return isFree ? "Бесплатно" : "Платно";
}

export function copyText(text: string) {
  if (typeof navigator === "undefined") return Promise.resolve(false);
  return navigator.clipboard
    .writeText(text)
    .then(() => true)
    .catch(() => false);
}
