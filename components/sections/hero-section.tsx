"use client";

import { Search, Sparkles } from "lucide-react";
import { motion } from "framer-motion";

import { Input } from "@/components/ui/input";

type HeroSectionProps = {
  search: string;
  onSearch: (value: string) => void;
  onQuickPick: (value: string) => void;
};

const quickTags = ["Код", "Видео", "Бесплатные", "Голос", "Сравнение"];

export function HeroSection({ search, onSearch, onQuickPick }: HeroSectionProps) {
  return (
    <section className="relative overflow-hidden rounded-[32px] border border-cyan/20 bg-white/5 px-6 py-12 shadow-neon backdrop-blur-2xl md:px-10 md:py-16">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(0,242,255,0.18),transparent_30%),radial-gradient(circle_at_80%_20%,rgba(255,0,245,0.16),transparent_22%)]" />
      <div className="relative grid gap-10 lg:grid-cols-[1.2fr_0.8fr] lg:items-end">
        <div className="space-y-6">
          <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-magenta/30 bg-magenta/10 px-4 py-2 text-xs uppercase tracking-[0.3em] text-magenta">
              <Sparkles className="h-4 w-4" />
              Премиум-каталог 2026
            </div>
            <h1 className="max-w-4xl text-4xl font-semibold leading-tight text-white text-glow md:text-6xl">
              Все нейросети мира в одном месте
            </h1>
          </motion.div>
          <p className="max-w-2xl text-base leading-8 text-white/72 md:text-lg">
            Исследуй флагманские AI-модели для текста, кода, изображений, видео, музыки и поиска.
            Сравнивай, фильтруй, сохраняй в избранное и тестируй демо-сценарии прямо в браузере.
          </p>
          <div className="relative max-w-3xl">
            <Search className="pointer-events-none absolute left-5 top-1/2 h-5 w-5 -translate-y-1/2 text-cyan/70" />
            <Input
              value={search}
              onChange={(event) => onSearch(event.target.value)}
              className="h-14 rounded-[22px] border-cyan/25 bg-[#050a14]/85 pl-14 pr-4 text-base shadow-[0_0_24px_rgba(0,242,255,0.18)]"
              placeholder="Найти модель, компанию, категорию или сценарий..."
            />
          </div>
          <div className="flex flex-wrap gap-3">
            {quickTags.map((tag) => (
              <button
                key={tag}
                onClick={() => onQuickPick(tag)}
                className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-white/70 transition hover:border-cyan/35 hover:bg-cyan/10 hover:text-cyan"
              >
                #{tag}
              </button>
            ))}
          </div>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1">
          {[
            ["20+", "реальных моделей"],
            ["6", "основных категорий"],
            ["100%", "статический деплой"],
            ["0 API", "для демо-режима"],
          ].map(([value, label]) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
              className="rounded-[28px] border border-white/10 bg-[#071018]/80 p-5 shadow-[0_0_24px_rgba(255,0,245,0.14)]"
            >
              <div className="text-3xl font-semibold text-cyan text-glow">{value}</div>
              <div className="mt-2 text-sm uppercase tracking-[0.28em] text-white/55">{label}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
