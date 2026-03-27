import type * as React from "react";

import { cn } from "@/lib/utils";

export function Badge({ className, children }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full border border-cyan/20 bg-cyan/10 px-3 py-1 text-[11px] uppercase tracking-[0.24em] text-cyan",
        className,
      )}
    >
      {children}
    </div>
  );
}
