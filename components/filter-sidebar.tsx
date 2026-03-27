"use client";

import { Filter, RotateCcw, Star } from "lucide-react";

import type { ModelType } from "@/lib/models";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";

type FilterSidebarProps = {
  types: ModelType[];
  companies: string[];
  selectedTypes: ModelType[];
  selectedCompanies: string[];
  onlyFree: boolean;
  favoritesCount: number;
  onToggleType: (value: ModelType) => void;
  onToggleCompany: (value: string) => void;
  onOnlyFreeChange: (value: boolean) => void;
  onReset: () => void;
};

export function FilterSidebar({
  types,
  companies,
  selectedTypes,
  selectedCompanies,
  onlyFree,
  favoritesCount,
  onToggleType,
  onToggleCompany,
  onOnlyFreeChange,
  onReset,
}: FilterSidebarProps) {
  return (
    <aside className="glass sticky top-6 h-fit rounded-[28px] p-5 shadow-neon">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-white">
            <Filter className="h-5 w-5 text-cyan" />
            <h2 className="font-semibold">Фильтры</h2>
          </div>
          <p className="mt-2 text-sm text-white/55">Собери свой стек под задачу за пару кликов.</p>
        </div>
        <Button variant="ghost" size="sm" onClick={onReset}>
          <RotateCcw className="h-4 w-4" />
          Сбросить
        </Button>
      </div>

      <div className="mt-6 space-y-6">
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm uppercase tracking-[0.28em] text-white/55">Тип модели</h3>
            <Badge>{selectedTypes.length || "Все"}</Badge>
          </div>
          <div className="grid gap-3">
            {types.map((type) => (
              <label key={type} className="flex cursor-pointer items-center gap-3 rounded-2xl border border-white/8 bg-white/5 px-4 py-3 text-sm text-white/75 transition hover:border-cyan/25 hover:bg-cyan/10">
                <Checkbox checked={selectedTypes.includes(type)} onCheckedChange={() => onToggleType(type)} />
                <span>{type}</span>
              </label>
            ))}
          </div>
        </section>

        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm uppercase tracking-[0.28em] text-white/55">Компания</h3>
            <Badge>{selectedCompanies.length || "Все"}</Badge>
          </div>
          <div className="max-h-72 space-y-3 overflow-auto pr-1 scrollbar-thin">
            {companies.map((company) => (
              <label key={company} className="flex cursor-pointer items-center gap-3 rounded-2xl border border-white/8 bg-white/5 px-4 py-3 text-sm text-white/75 transition hover:border-magenta/25 hover:bg-magenta/10">
                <Checkbox checked={selectedCompanies.includes(company)} onCheckedChange={() => onToggleCompany(company)} />
                <span>{company}</span>
              </label>
            ))}
          </div>
        </section>

        <section className="rounded-[24px] border border-cyan/20 bg-cyan/10 p-4">
          <label className="flex cursor-pointer items-start gap-3">
            <Checkbox checked={onlyFree} onCheckedChange={(checked) => onOnlyFreeChange(Boolean(checked))} />
            <div>
              <div className="font-medium text-cyan">Только бесплатные модели</div>
              <p className="mt-1 text-sm leading-6 text-white/60">Показывать только модели с бесплатным тарифом или пробным режимом.</p>
            </div>
          </label>
        </section>

        <section className="rounded-[24px] border border-magenta/20 bg-magenta/10 p-4">
          <div className="flex items-center gap-3 text-white">
            <Star className="h-5 w-5 text-magenta" />
            <div>
              <div className="font-medium">Избранное</div>
              <p className="text-sm text-white/60">Сохранено моделей: {favoritesCount}</p>
            </div>
          </div>
        </section>
      </div>
    </aside>
  );
}
