export function SiteFooter() {
  return (
    <footer className="glass rounded-[32px] px-6 py-8 shadow-neon md:px-8">
      <div className="grid gap-8 md:grid-cols-[1.2fr_0.8fr_0.8fr_1fr]">
        <div>
          <div className="text-lg font-semibold uppercase tracking-[0.35em] text-cyan text-glow">AI Catalog</div>
          <p className="mt-4 max-w-md text-sm leading-7 text-white/62">
            Премиум-каталог нейросетей для тех, кто выбирает лучший AI-стек: от текста и кода до видео, музыки и голосовых агентов.
          </p>
        </div>
        <div>
          <div className="text-sm uppercase tracking-[0.28em] text-white/55">Разделы</div>
          <ul className="mt-4 space-y-3 text-sm text-white/68">
            <li>Тренды сегодня</li>
            <li>Новые модели</li>
            <li>Сравнение</li>
            <li>Избранное</li>
          </ul>
        </div>
        <div>
          <div className="text-sm uppercase tracking-[0.28em] text-white/55">Категории</div>
          <ul className="mt-4 space-y-3 text-sm text-white/68">
            <li>Текст и код</li>
            <li>Изображения</li>
            <li>Видео</li>
            <li>Аудио и голос</li>
          </ul>
        </div>
        <div>
          <div className="text-sm uppercase tracking-[0.28em] text-white/55">Статус проекта</div>
          <p className="mt-4 text-sm leading-7 text-white/62">
            Полностью клиентский демо-каталог. Сайт собран под статический экспорт и деплой на Render Free без серверной части.
          </p>
        </div>
      </div>
      <div className="mt-8 border-t border-white/10 pt-6 text-sm text-white/45">© 2026 AI Catalog. Сделано в киберпанк-стиле для быстрого деплоя и красивой витрины моделей.</div>
    </footer>
  );
}
