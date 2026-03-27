"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

import type { ModelCatalogItem, ModelType } from "@/lib/models";

type CatalogState = {
  search: string;
  selectedTypes: ModelType[];
  selectedCompanies: string[];
  onlyFree: boolean;
  favorites: string[];
  activeModelId: string | null;
  modalOpen: boolean;
  setSearch: (value: string) => void;
  toggleType: (value: ModelType) => void;
  toggleCompany: (value: string) => void;
  setOnlyFree: (value: boolean) => void;
  toggleFavorite: (id: string) => void;
  openModal: (id: string) => void;
  closeModal: () => void;
  resetFilters: () => void;
};

export const useCatalogStore = create<CatalogState>()(
  persist(
    (set) => ({
      search: "",
      selectedTypes: [],
      selectedCompanies: [],
      onlyFree: false,
      favorites: [],
      activeModelId: null,
      modalOpen: false,
      setSearch: (search) => set({ search }),
      toggleType: (value) =>
        set((state) => ({
          selectedTypes: state.selectedTypes.includes(value)
            ? state.selectedTypes.filter((item) => item !== value)
            : [...state.selectedTypes, value],
        })),
      toggleCompany: (value) =>
        set((state) => ({
          selectedCompanies: state.selectedCompanies.includes(value)
            ? state.selectedCompanies.filter((item) => item !== value)
            : [...state.selectedCompanies, value],
        })),
      setOnlyFree: (onlyFree) => set({ onlyFree }),
      toggleFavorite: (id) =>
        set((state) => ({
          favorites: state.favorites.includes(id)
            ? state.favorites.filter((item) => item !== id)
            : [...state.favorites, id],
        })),
      openModal: (id) => set({ activeModelId: id, modalOpen: true }),
      closeModal: () => set({ modalOpen: false }),
      resetFilters: () => set({ search: "", selectedCompanies: [], selectedTypes: [], onlyFree: false }),
    }),
    {
      name: "ai-catalog-store",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ favorites: state.favorites }),
    },
  ),
);

export function findActiveModel(models: ModelCatalogItem[], modelId: string | null) {
  return models.find((item) => item.id === modelId) ?? null;
}
