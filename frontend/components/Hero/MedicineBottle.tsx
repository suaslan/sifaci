"use client";

import { motion } from "motion/react";

type MedicineBottleProps = {
  reducedMotion: boolean;
  entranceReady: boolean;
};

const bottlePills = [
  { className: "bottle-pill bottle-pill--round", x: 22, y: 104, rotate: -12 },
  { className: "bottle-pill bottle-pill--capsule", x: 58, y: 122, rotate: 32 },
  { className: "bottle-pill bottle-pill--oval", x: 31, y: 145, rotate: 18 },
  { className: "bottle-pill bottle-pill--round", x: 69, y: 158, rotate: 8 },
] as const;

export function MedicineBottle({ reducedMotion, entranceReady }: MedicineBottleProps) {
  return (
    <motion.div
      className="medicine-bottle"
      initial={false}
      animate={
        entranceReady && !reducedMotion
          ? { opacity: [0, 1], x: [34, 0], y: [18, 0], scale: [0.9, 1] }
          : { opacity: 1, x: 0, y: 0, scale: 1 }
      }
      transition={{
        duration: reducedMotion ? 0 : 0.9,
        delay: reducedMotion ? 0 : 0.72,
        ease: [0.22, 1, 0.36, 1],
      }}
      aria-hidden="true"
    >
      <div className="medicine-bottle__cap">
        <span />
        <span />
        <span />
        <span />
        <span />
      </div>
      <div className="medicine-bottle__neck" />
      <div className="medicine-bottle__body">
        <span className="medicine-bottle__shine" />
        <span className="medicine-bottle__base" />
        {bottlePills.map((pill, index) => (
          <motion.span
            key={`${pill.className}-${pill.x}`}
            className={pill.className}
            style={{ left: pill.x, top: pill.y, rotate: pill.rotate }}
            animate={reducedMotion ? undefined : { y: [0, index % 2 ? -3 : 2, 0] }}
            transition={{
              duration: 3.8 + index * 0.35,
              delay: 1.6 + index * 0.18,
              repeat: Number.POSITIVE_INFINITY,
              ease: "easeInOut",
            }}
          />
        ))}
        <div className="medicine-bottle__label">
          <span>KRİSTAL FORMÜL</span>
          <strong>Şifacı AI</strong>
          <small>RAG • DOĞRULANMIŞ VERİ</small>
        </div>
      </div>
    </motion.div>
  );
}
