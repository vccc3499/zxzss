"use client";

import { useEffect, useMemo, useState } from "react";
import { Copy, ExternalLink, Send, Sparkles, WandSparkles } from "lucide-react";

import type { ModelCatalogItem } from "@/lib/models";
import { copyText } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const sampleOutputs: Record<string, string[]> = {
  "Текст": [
    "Готово. Сформировал чёткий ответ с акцентом на стратегию, риски и пошаговый план внедрения.",
    "Модель предлагает три сценария: быстрый запуск, умеренный рост и агрессивную экспансию.",
    "Добавлен финальный блок с KPI, бюджетом и каналами привлечения клиентов.",
  ],
  "Код": [
    "Сгенерирован каркас проекта: App Router, Zustand-store, UI-компоненты и static export.",
    "Добавлены рекомендации по архитектуре, разбиению файлов и проверке production-сборки.",
    "Подготовлен финальный список команд для локального запуска и деплоя на Render Free.",
  ],
  "Изображения": [
    "Промт уточнён: cinematic cyberpunk, premium neon glassmorphism, high detail, electric blue and magenta glow.",
    "Сформирован art direction: глубокий фон, стеклянные панели, контрастные подсветки, мягкий bloom.",
    "Добавлены негативные ограничения: без мусора в кадре, без перегруза типографикой, clean premium layout.",
  ],
  "Видео": [
    "Собран синопсис ролика: пролёт камеры, reveal интерфейса, акцент на карточках моделей и CTA.",
    "Генерация имитирует сториборд: 5 сцен, неоновые переходы, тексты для титров и финальный логотип.",
    "Добавлен production note: slow dolly-in, volumetric light, premium cyberpunk city ambience.",
  ],
  "Аудио": [
    "Подготовлен джингл: synthwave groove, уверенный темп, электронный бас и фирменный hook.",
    "Добавлены варианты аранжировки: короткая заставка, full loop и версия под рекламный ролик.",
    "Финальная подача: современный неоновый звук с чистым продакшном и ярким припевом.",
  ],
  "Поиск": [
    "Собрана краткая сводка по рынку: ключевые игроки, новые релизы и практические выводы.",
    "Выделены 3 источника роста и 4 риска при выборе модели для бизнеса.",
    "Подготовлен короткий decision memo с рекомендацией, куда смотреть дальше.",
  ],
};

type ModelModalProps = {
  model: ModelCatalogItem | null;
  open: boolean;
  onOpenChange: (value: boolean) => void;
};

export function ModelModal({ model, open, onOpenChange }: ModelModalProps) {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; text: string }[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [copyState, setCopyState] = useState("Скопировать промт");

  useEffect(() => {
    if (!model) return;
    setPrompt(model.promptExample);
    setMessages([
      {
        role: "assistant",
        text: `Демо-режим активирован для ${model.name}. Это клиентская имитация без вызова API: можно протестировать UX, формулировки и подачу ответа.`,
      },
    ]);
    setIsGenerating(false);
  }, [model]);

  const outputPack = useMemo(() => {
    if (!model) return [];
    return sampleOutputs[model.type] ?? sampleOutputs["Текст"];
  }, [model]);

  const handleGenerate = async () => {
    if (!model || !prompt.trim() || isGenerating) return;
    setMessages((prev) => [...prev, { role: "user", text: prompt }]);
    setIsGenerating(true);
    await new Promise((resolve) => window.setTimeout(resolve, 900));
    const text = `${outputPack.join(" ")}\n\nЛучше всего подходит для: ${model.bestFor.join(", ")}.\n\nСильные стороны: ${model.strengths.join(", ")}.`;
    setMessages((prev) => [...prev, { role: "assistant", text }]);
    setIsGenerating(false);
  };

  const handleCopy = async () => {
    if (!prompt) return;
    const success = await copyText(prompt);
    setCopyState(success ? "Промт скопирован" : "Не удалось скопировать");
    window.setTimeout(() => setCopyState("Скопировать промт"), 1800);
  };

  if (!model) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[92vh] overflow-hidden p-0">
        <div className="grid max-h-[92vh] gap-0 lg:grid-cols-[0.92fr_1.08fr]">
          <div className="border-b border-white/10 bg-[#050914] p-6 lg:border-b-0 lg:border-r">
            <DialogHeader>
              <div className="mb-4 flex items-center gap-3">
                <div
                  className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/20 text-lg font-semibold text-white"
                  style={{ background: `linear-gradient(135deg, ${model.accentFrom}, ${model.accentTo})` }}
                >
                  {model.companyShort}
                </div>
                <div>
                  <DialogTitle>{model.name}</DialogTitle>
                  <DialogDescription>{model.company} · {model.type}</DialogDescription>
                </div>
              </div>
            </DialogHeader>

            <p className="text-sm leading-7 text-white/72">{model.longDescription}</p>

            <div className="mt-6 grid gap-4">
              <div className="rounded-[24px] border border-cyan/20 bg-cyan/10 p-4">
                <div className="text-xs uppercase tracking-[0.28em] text-cyan">Почему в тренде</div>
                <div className="mt-2 text-sm leading-7 text-white/75">{model.releaseNote}</div>
              </div>
              <div className="rounded-[24px] border border-white/10 bg-white/5 p-4">
                <div className="text-xs uppercase tracking-[0.28em] text-white/55">Лучше всего для</div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {model.bestFor.map((item) => (
                    <span key={item} className="rounded-full border border-white/10 bg-black/20 px-3 py-1 text-xs text-white/70">
                      {item}
                    </span>
                  ))}
                </div>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <Button variant="secondary" onClick={handleCopy}>
                  <Copy className="h-4 w-4" />
                  {copyState}
                </Button>
                <a href={model.url} target="_blank" rel="noreferrer" className="inline-flex">
                  <Button className="w-full" variant="magenta">
                    <ExternalLink className="h-4 w-4" />
                    На сайт модели
                  </Button>
                </a>
              </div>
            </div>
          </div>

          <div className="flex min-h-[560px] flex-col bg-[#060913]/95">
            <div className="border-b border-white/10 p-6">
              <div className="flex items-center gap-3 text-sm text-white/65">
                <WandSparkles className="h-4 w-4 text-cyan" />
                Имитация ответа на клиенте. Подходит для демонстрации UX, но не требует API.
              </div>
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                className="mt-4 min-h-28 w-full rounded-[22px] border border-cyan/20 bg-black/20 px-4 py-4 text-sm leading-7 text-white outline-none ring-0 placeholder:text-white/35"
                placeholder="Опиши задачу, которую хочешь проверить на этой модели"
              />
              <div className="mt-4 flex flex-wrap gap-3">
                <Button onClick={handleGenerate} disabled={isGenerating || !prompt.trim()}>
                  <Send className="h-4 w-4" />
                  {isGenerating ? "Генерация..." : "Попробовать модель"}
                </Button>
                <Button variant="ghost" onClick={() => setPrompt(model.promptExample)}>
                  <Sparkles className="h-4 w-4" />
                  Подставить демо-промт
                </Button>
              </div>
            </div>

            <div className="flex-1 space-y-4 overflow-auto p-6 scrollbar-thin">
              {messages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={`max-w-[92%] rounded-[24px] border px-5 py-4 text-sm leading-7 ${
                    message.role === "user"
                      ? "ml-auto border-cyan/30 bg-cyan/10 text-cyan"
                      : "border-white/10 bg-white/5 text-white/78"
                  }`}
                >
                  {message.text}
                </div>
              ))}
              {isGenerating ? (
                <div className="max-w-[92%] rounded-[24px] border border-magenta/20 bg-magenta/10 px-5 py-4 text-sm text-white/80">
                  Генерирую ответ... подготавливаю структуру, ключевые блоки и финальную подачу.
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
