"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, Copy, Heart, SearchCode, Star } from "lucide-react";

import { FilterSidebar } from "@/components/filter-sidebar";
import { ModelCard } from "@/components/model-card";
import { ModelModal } from "@/components/model-modal";
import { ParticlesBackground } from "@/components/particles-background";
import { SiteFooter } from "@/components/site-footer";
import { ComparisonTable } from "@/components/sections/comparison-table";
import { HeroSection } from "@/components/sections/hero-section";
import { TrendsSection } from "@/components/sections/trends-section";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { QUICK_PROMPTS, AI_MODELS, FEATURED_COMPARE_IDS, MODEL_TYPES } from "@/lib/models";
import { useDebouncedValue } from "@/hooks/use-debounced-value";
import { findActiveModel, useCatalogStore } from "@/store/catalog-store";
import { copyText } from "@/lib/utils";

export default function HomePage() {
  const {
    search,
    selectedTypes,
    selectedCompanies,
    onlyFree,
    favorites,
    modalOpen,
    activeModelId,
    setSearch,
    toggleType,
    toggleCompany,
    setOnlyFree,
    toggleFavorite,
    openModal,
    closeModal,
    resetFilters,
  } = useCatalogStore();

  const [mounted, setMounted] = useState(false);
  const [copiedPrompt, setCopiedPrompt] = useState<string | null>(null);
  const debouncedSearch = useDebouncedValue(search, 280);

  useEffect(() => {
    setMounted(true);
  }, []);

  const companies = useMemo(() => Array.from(new Set(AI_MODELS.map((model) => model.company))).sort(), []);

  const filteredModels = useMemo(() => {
    const needle = debouncedSearch.trim().toLowerCase();

    return AI_MODELS.filter((model) => {
      const matchesSearch =
        !needle ||
        [model.name, model.company, model.description, model.type, model.tags.join(" "), model.bestFor.join(" ")]
          .join(" ")
          .toLowerCase()
          .includes(needle);
      const matchesType = selectedTypes.length === 0 || selectedTypes.includes(model.type);
      const matchesCompany = selectedCompanies.length === 0 || selectedCompanies.includes(model.company);
      const matchesPrice = !onlyFree || model.isFree;
      return matchesSearch && matchesType && matchesCompany && matchesPrice;
    }).sort((a, b) => b.trendScore - a.trendScore);
  }, [debouncedSearch, onlyFree, selectedCompanies, selectedTypes]);

  const trendingModels = useMemo(() => [...AI_MODELS].sort((a, b) => b.trendScore - a.trendScore).slice(0, 4), []);
  const newModels = useMemo(() => AI_MODELS.filter((item) => item.isNew).slice(0, 4), []);
  const bestByCategory = useMemo(() => {
    const map = new Map<string, typeof AI_MODELS[number]>();
    for (const item of AI_MODELS) {
      if (!map.has(item.type)) map.set(item.type, item);
    }
    return Array.from(map.values()).slice(0, 4);
  }, []);
  const compareModels = useMemo(() => AI_MODELS.filter((item) => FEATURED_COMPARE_IDS.includes(item.id)), []);
  const activeModel = findActiveModel(AI_MODELS, activeModelId);

  const handleCopyPrompt = async (prompt: string) => {
    const ok = await copyText(prompt);
    setCopiedPrompt(ok ? prompt : null);
    window.setTimeout(() => setCopiedPrompt(null), 1800);
  };

  return (
    <main className="relative min-h-screen overflow-hidden px-3 py-3 text-white md:px-5 md:py-5">
      <ParticlesBackground />
      <div className="relative z-10 mx-auto flex w-full max-w-[1560px] flex-col gap-6">
        <HeroSection search={search} onSearch={setSearch} onQuickPick={setSearch} />

        <section className="grid gap-6 xl:grid-cols-[320px_1fr]">
          <FilterSidebar
            types={MODEL_TYPES}
            companies={companies}
            selectedTypes={selectedTypes}
            selectedCompanies={selectedCompanies}
            onlyFree={onlyFree}
            favoritesCount={favorites.length}
            onToggleType={toggleType}
            onToggleCompany={toggleCompany}
            onOnlyFreeChange={setOnlyFree}
            onReset={resetFilters}
          />

          <div className="space-y-6">
            <div className="glass rounded-[30px] p-5 shadow-neon md:p-6">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                <div>
                  <Badge>Каталог моделей</Badge>
                  <h2 className="mt-4 text-3xl font-semibold text-white md:text-4xl">Подборка под твой стек</h2>
                  <p className="mt-3 text-white/65">
                    Найдено моделей: <span className="text-cyan">{filteredModels.length}</span> из <span className="text-white">{AI_MODELS.length}</span>
                  </p>
                </div>
                <div className="flex flex-wrap gap-3">
                  <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-white/70">Избранное: {mounted ? favorites.length : 0}</div>
                  <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-white/70">Бесплатные: {AI_MODELS.filter((item) => item.isFree).length}</div>
                </div>
              </div>
            </div>

            <div className="grid gap-5 md:grid-cols-2 2xl:grid-cols-3">
              <AnimatePresence>
                {filteredModels.map((model) => (
                  <ModelCard
                    key={model.id}
                    model={model}
                    isFavorite={favorites.includes(model.id)}
                    onOpen={openModal}
                    onFavorite={toggleFavorite}
                  />
                ))}
              </AnimatePresence>
            </div>

            {filteredModels.length === 0 ? (
              <div className="glass rounded-[30px] p-8 text-center shadow-neon">
                <SearchCode className="mx-auto h-10 w-10 text-cyan" />
                <h3 className="mt-4 text-2xl font-semibold text-white">Ничего не найдено</h3>
                <p className="mt-3 text-white/62">Сними часть фильтров или измени поисковый запрос. Каталог обновится мгновенно.</p>
                <Button className="mt-6" onClick={resetFilters}>Сбросить фильтры</Button>
              </div>
            ) : null}
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <TrendsSection
            title="Тренды сегодня"
            eyebrow="Тренды"
            description="Модели, которые чаще всего обсуждают, интегрируют и сравнивают прямо сейчас."
            items={trendingModels}
          />
          <div className="glass rounded-[32px] p-6 shadow-magenta">
            <Badge>Быстрые действия</Badge>
            <h2 className="mt-4 text-3xl font-semibold text-white">Готовые промты</h2>
            <p className="mt-3 text-white/65">Скопируй вопрос и сразу проверяй модели на похожем сценарии.</p>
            <div className="mt-6 grid gap-3">
              {QUICK_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => handleCopyPrompt(prompt)}
                  className="rounded-[22px] border border-white/10 bg-white/5 px-4 py-4 text-left text-sm leading-7 text-white/78 transition hover:border-cyan/30 hover:bg-cyan/10"
                >
                  <div className="flex items-start justify-between gap-3">
                    <span>{prompt}</span>
                    <Copy className="mt-1 h-4 w-4 shrink-0 text-cyan" />
                  </div>
                  {copiedPrompt === prompt ? <div className="mt-2 text-xs uppercase tracking-[0.28em] text-cyan">Скопировано</div> : null}
                </button>
              ))}
            </div>
          </div>
        </section>

        <TrendsSection
          title="Новые модели"
          eyebrow="Fresh Drop"
          description="Свежие релизы и заметные обновления, которые уже влияют на рынок AI-продуктов."
          items={newModels}
        />

        <TrendsSection
          title="Лучшие по категориям"
          eyebrow="Best Picks"
          description="По одной сильной модели на каждую категорию, если нужен быстрый ориентир без долгого выбора."
          items={bestByCategory}
        />

        <ComparisonTable items={compareModels} />

        <section className="glass overflow-hidden rounded-[32px] p-6 shadow-neon md:p-8">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <Badge>Почему это удобно</Badge>
              <h2 className="mt-4 text-3xl font-semibold text-white md:text-4xl">Премиум-витрина для выбора AI-стека</h2>
              <p className="mt-3 max-w-2xl text-white/65">
                Сайт работает целиком на клиенте, хранит избранное в localStorage, а демо-режим не требует серверных ключей. Это удобно для лендинга, каталога, showcase-проекта и быстрого деплоя на бесплатный Render.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              {[
                ["100% client-side", "Демо, поиск, фильтры и избранное работают без backend."],
                ["Render Free Ready", "Статический экспорт, минимальная нагрузка, быстрый деплой."],
                ["Русский интерфейс", "Вся витрина, мета-теги и UX полностью на русском."],
                ["Cyberpunk Premium", "Дорогой визуал с неоном, стеклом и плавной анимацией."],
              ].map(([title, text]) => (
                <motion.div key={title} whileHover={{ y: -4 }} className="rounded-[24px] border border-white/10 bg-white/5 p-4">
                  <div className="flex items-center gap-2 text-white">
                    <Star className="h-4 w-4 text-cyan" />
                    <div className="font-medium">{title}</div>
                  </div>
                  <p className="mt-2 text-sm leading-7 text-white/62">{text}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <SiteFooter />
      </div>

      <ModelModal model={activeModel} open={modalOpen} onOpenChange={(value) => (value ? undefined : closeModal())} />
    </main>
  );
}
