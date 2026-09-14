"use client";

import {
  motion,
  useMotionValue,
  useReducedMotion,
  useSpring,
  useTransform,
} from "motion/react";
import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import { MedicineAnswerCard } from "@/components/Answer/MedicineAnswerCard";
import { Header } from "@/components/Header";
import { MedicineBox } from "@/components/Hero/MedicineBox";
import { MedicineSearch } from "@/components/Search/MedicineSearch";
import { askMedicineAssistant } from "@/lib/api";
import type { MedicineAnswer } from "@/types/medicine";

const premiumEase = [0.22, 1, 0.36, 1] as const;

export function Hero() {
  const reducedMotion = Boolean(useReducedMotion());
  const pointerX = useMotionValue(0);
  const pointerY = useMotionValue(0);
  const shadowX = useSpring(pointerX, { stiffness: 70, damping: 22, mass: 0.8 });
  const shadowY = useSpring(pointerY, { stiffness: 70, damping: 22, mass: 0.8 });
  const primaryShadowX = useTransform(shadowX, [-1, 1], [-28, 28]);
  const primaryShadowY = useTransform(shadowY, [-1, 1], [-14, 14]);
  const secondaryShadowX = useTransform(shadowX, [-1, 1], [18, -18]);
  const secondaryShadowY = useTransform(shadowY, [-1, 1], [10, -10]);
  const activeRequest = useRef<AbortController | null>(null);
  const [activeQuestion, setActiveQuestion] = useState("");
  const [answer, setAnswer] = useState<MedicineAnswer | null>(null);
  const [responseTimeMs, setResponseTimeMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [entranceReady, setEntranceReady] = useState(false);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => setEntranceReady(true));
    return () => {
      window.cancelAnimationFrame(frame);
      activeRequest.current?.abort();
    };
  }, []);

  const moveAmbientShadow = (event: ReactPointerEvent<HTMLElement>) => {
    if (reducedMotion) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    pointerX.set(((event.clientX - bounds.left) / bounds.width) * 2 - 1);
    pointerY.set(((event.clientY - bounds.top) / bounds.height) * 2 - 1);
  };

  const handleSearch = async (question: string) => {
    activeRequest.current?.abort();
    const controller = new AbortController();
    const startedAt = performance.now();
    activeRequest.current = controller;
    setActiveQuestion(question);
    setAnswer(null);
    setResponseTimeMs(null);
    setError(null);
    setIsLoading(true);

    try {
      const response = await askMedicineAssistant(question, controller.signal);
      if (activeRequest.current === controller) {
        setAnswer(response);
        setResponseTimeMs(performance.now() - startedAt);
      }
    } catch (requestError) {
      if (requestError instanceof Error && requestError.name === "AbortError") return;
      if (activeRequest.current === controller) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Şifacı AI şu anda yanıt oluşturamadı.",
        );
      }
    } finally {
      if (activeRequest.current === controller) {
        setIsLoading(false);
      }
    }
  };

  const reveal = (delay: number, distance = 20) => ({
    initial: false as const,
    animate:
      entranceReady && !reducedMotion
        ? { opacity: [0, 1], y: [distance, 0] }
        : { opacity: 1, y: 0 },
    transition: {
      duration: reducedMotion ? 0 : 0.82,
      delay: reducedMotion ? 0 : delay,
      ease: premiumEase,
    },
  });

  return (
    <section
      id="top"
      className="relative min-h-svh overflow-hidden border-b border-red-900/10"
      onPointerMove={moveAmbientShadow}
    >
      <motion.span
        className="hero-orb top-[8%] left-[-8rem] h-96 w-96 bg-red-200/55"
        style={reducedMotion ? undefined : { x: primaryShadowX, y: primaryShadowY }}
        aria-hidden="true"
      />
      <motion.span
        className="hero-orb right-[-10rem] top-[30%] h-[28rem] w-[28rem] bg-amber-200/45"
        style={reducedMotion ? undefined : { x: secondaryShadowX, y: secondaryShadowY }}
        aria-hidden="true"
      />
      <div className="hero-grid" aria-hidden="true" />

      <Header />

      <div className="relative z-10 mx-auto flex w-full max-w-[1220px] flex-col items-center px-5 pt-4 pb-16 text-center sm:px-8 lg:pt-6 lg:pb-20">
        <motion.div
          initial={false}
          animate={
            entranceReady && !reducedMotion
              ? { opacity: [0, 1], scale: [0.94, 1], y: [24, 0] }
              : { opacity: 1, scale: 1, y: 0 }
          }
          transition={{ duration: reducedMotion ? 0 : 1, delay: reducedMotion ? 0 : 0.28, ease: premiumEase }}
          className="relative mt-2 flex h-[25rem] w-full items-center justify-center sm:h-[28rem]"
        >
          <MedicineBox />
        </motion.div>

        <motion.div {...reveal(0.42, 18)} className="relative z-30 -mt-16 w-full max-w-[64rem] sm:-mt-20">
          <MedicineSearch onSearch={handleSearch} isLoading={isLoading} />
          <MedicineAnswerCard
            question={activeQuestion}
            loading={isLoading}
            result={answer}
            error={error}
            responseTimeMs={responseTimeMs}
            onSuggestionSelect={handleSearch}
          />
        </motion.div>
      </div>
    </section>
  );
}
