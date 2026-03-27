"use client";

import { Heart, Sparkles, Star } from "lucide-react";
import { motion } from "framer-motion";

import type { ModelCatalogItem } from "@/lib/models";
import { formatPriceLabel } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

type ModelCardProps = {
  model: ModelCatalogItem;
  isFavorite: boolean;
  onOpen: (id: string) => void;
  onFavorite: (id: string) => void;
};

export function ModelCard({ model, isFavorite, onOpen, onFavorite }: ModelCardProps) {
  return (
    <motion.article
      layout
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className="group glass overflow-hidden rounded-[30px] shadow-[0_0_34px_rgba(0,242,255,0.12)]"
    >
      <div
        className="relative h-44 overflow-hidden border-b border-white/10"
        style={{ background: `linear-gradient(135deg, ${model.accentFrom}, ${model.accentTo})` }}
      >
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(255,255,255,0.35),transparent_24%),linear-gradient(135deg,rgba(3,6,20,0.18),rgba(3,6,20,0.7))]" />
        <div className="absolute left-5 top-5 flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/30 bg-black/20 text-lg font-semibold text-white shadow-[0_0_24px_rgba(255,255,255,0.18)]">
            {model.companyShort}
          </div>
          <div>
            <div className="text-xs uppercase tracking-[0.28em] text-white/70">{model.company}</div>
            <div className="text-lg font-semibold text-white">{model.name}</div>
          </div>
        </div>
        <div className="absolute bottom-5 left-5 right-5 flex items-end justify-between gap-3">
          <Badge className="border-white/20 bg-black/20 text-white">{model.heroBadge}</Badge>
          <div className="rounded-full border border-white/20 bg-black/20 px-3 py-1 text-xs uppercase tracking-[0.24em] text-white/85">
            {model.type}
          </div>
        </div>
      </div>

      <div className="space-y-4 p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-sm leading-7 text-white/72">{model.description}</p>
          </div>
          <button
            onClick={() => onFavorite(model.id)}
            className="rounded-full border border-white/10 bg-white/5 p-3 text-white/60 transition hover:border-magenta/40 hover:text-magenta"
            aria-label="Добавить в избранное"
          >
            <Heart className={`h-4 w-4 ${isFavorite ? "fill-current text-magenta" : ""}`} />
          </button>
        </div>

        <div className="flex flex-wrap gap-2">
          {model.tags.slice(0, 3).map((tag) => (
            <span key={tag} className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/55">
              {tag}
            </span>
          ))}
        </div>

        <div className="flex items-center justify-between gap-3 border-y border-white/8 py-3">
          <div className="flex items-center gap-2 text-sm text-white/70">
            <Star className="h-4 w-4 fill-current text-cyan" />
            {model.rating.toFixed(1)} / 5.0
          </div>
          <div className="text-sm text-white/55">{formatPriceLabel(model.isFree)}</div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <Button onClick={() => onOpen(model.id)}>
            <Sparkles className="h-4 w-4" />
            Попробовать
          </Button>
          <Button variant="secondary" onClick={() => onFavorite(model.id)}>
            <Heart className={`h-4 w-4 ${isFavorite ? "fill-current text-magenta" : ""}`} />
            В избранное
          </Button>
        </div>
      </div>
    </motion.article>
  );
}
