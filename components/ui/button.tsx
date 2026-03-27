import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-full border text-sm font-medium transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "border-cyan/40 bg-cyan/10 text-cyan shadow-neon hover:-translate-y-0.5 hover:bg-cyan/15",
        secondary: "border-white/10 bg-white/5 text-white/80 hover:bg-white/10",
        ghost: "border-transparent bg-transparent text-white/70 hover:border-cyan/30 hover:bg-cyan/10 hover:text-cyan",
        magenta: "border-magenta/40 bg-magenta/10 text-magenta shadow-magenta hover:-translate-y-0.5 hover:bg-magenta/15",
      },
      size: {
        default: "h-11 px-5",
        sm: "h-9 px-4 text-xs",
        lg: "h-12 px-6 text-base",
        icon: "h-11 w-11",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, ...props }, ref) => {
  return <button className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
});
Button.displayName = "Button";

export { Button, buttonVariants };
