"use client";

import type { ModelCatalogItem } from "@/lib/models";
import { Badge } from "@/components/ui/badge";

type ComparisonTableProps = {
  items: ModelCatalogItem[];
};

export function ComparisonTable({ items }: ComparisonTableProps) {
  return (
    <section className="glass overflow-hidden rounded-[32px] shadow-neon">
      <div className="border-b border-white/10 px-6 py-6 md:px-8">
        <Badge>Сравнение моделей</Badge>
        <h2 className="mt-4 text-3xl font-semibold text-white md:text-4xl">Сравнение флагманов</h2>
        <p className="mt-3 max-w-2xl text-white/65">Быстрая сводка по моделям, которые чаще всего сравнивают при выборе стека на 2026 год.</p>
      </div>
      <div className="overflow-auto scrollbar-thin">
        <table className="min-w-full text-left text-sm text-white/72">
          <thead className="bg-white/5 text-xs uppercase tracking-[0.28em] text-white/55">
            <tr>
              <th className="px-6 py-4 md:px-8">Модель</th>
              <th className="px-6 py-4">Тип</th>
              <th className="px-6 py-4">Сильна в</th>
              <th className="px-6 py-4">Лучше всего для</th>
              <th className="px-6 py-4">Доступ</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.id} className="border-t border-white/8 align-top">
                <td className="px-6 py-5 md:px-8">
                  <div className="font-semibold text-white">{item.name}</div>
                  <div className="mt-1 text-white/50">{item.company}</div>
                </td>
                <td className="px-6 py-5">{item.type}</td>
                <td className="px-6 py-5">{item.strengths.join(", ")}</td>
                <td className="px-6 py-5">{item.bestFor.join(", ")}</td>
                <td className="px-6 py-5">{item.pricingHint}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
