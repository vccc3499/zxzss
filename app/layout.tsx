import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";

import "./globals.css";

const fontUi = Inter({ subsets: ["latin", "cyrillic"], variable: "--font-ui" });
const fontMono = JetBrains_Mono({ subsets: ["latin", "cyrillic"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Каталог нейросетей — все нейросети мира в одном месте",
  description: "Премиум-каталог моделей ИИ: текст, код, изображения, видео, аудио, поиск и сравнение в одном киберпанк-интерфейсе.",
  keywords: ["нейросети", "каталог ИИ", "GPT", "Claude", "Gemini", "Flux", "Midjourney", "Runway"],
  openGraph: {
    title: "Все нейросети мира в одном месте",
    description: "Премиум-каталог нейросетей с поиском, фильтрами, сравнением и демо-режимом.",
    type: "website",
    locale: "ru_RU",
  },
  twitter: {
    card: "summary_large_image",
    title: "Все нейросети мира в одном месте",
    description: "Премиум-каталог моделей ИИ в киберпанк-стиле.",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ru" className="dark">
      <body className={`${fontUi.variable} ${fontMono.variable} font-sans`}>{children}</body>
    </html>
  );
}
