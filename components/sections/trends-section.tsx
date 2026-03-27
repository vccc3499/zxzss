"use client";

import { TrendingUp } from "lucide-react";

import type { ModelCatalogItem } from "@/lib/models";
import { Badge } from "@/components/ui/badge";

type TrendsSectionProps = {
  title: string;
  eyebrow: string;
  description: string;
  items: ModelCatalogItem[];
};

export function TrendsSection({ title, eyebrow, description, items }: TrendsSectionProps) {
  return (
    <section className="space-y-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <Badge>{eyebrow}</Badge>
          <h2 className="mt-4 text-3xl font-semibold text-white md:text-4xl">{title}</h2>
          <p className="mt-3 max-w-2xl text-white/65">{description}</p>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {items.map((item) => (
          <div key={item.id} className="glass rounded-[28px] p-5 shadow-neon">
            <div className="flex items-center justify-between gap-3">
              <div className="rounded-full border border-white/15 px-3 py-1 text-xs uppercase tracking-[0.28em] text-white/60">
                {item.company}
              </div>
              <div className="flex items-center gap-2 text-sm text-cyan">
                <TrendingUp className="h-4 w-4" />
                {item.trendScore}
              </div>
            </div>
            <h3 className="mt-4 text-xl font-semibold text-white">{item.name}</h3>
            <p className="mt-3 text-sm leading-7 text-white/68">{item.description}</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {item.tags.slice(0, 2).map((tag) => (
                <span key={tag} className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/55">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
