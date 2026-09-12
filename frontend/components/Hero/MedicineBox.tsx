"use client";

import {
  motion,
  useMotionValue,
  useReducedMotion,
  useSpring,
  useTransform,
} from "motion/react";
import type { PointerEvent as ReactPointerEvent } from "react";

import { FloatingPill, type FloatingPillProps } from "@/components/Hero/FloatingPill";

type AnimatedPill = Omit<FloatingPillProps, "reducedMotion" | "entranceReady">;

const animatedPills: AnimatedPill[] = [
  {
    id: "orbit-round",
    variant: "round",
    x: 190,
    y: -112,
    rotation: -20,
    scale: 0.82,
    delay: 0,
    floatDistance: 0,
    cycleDuration: 12.8,
    trajectory: [
      { x: 96, y: -30 },
      { x: 172, y: -132 },
      { x: 246, y: -28 },
      { x: 202, y: 88 },
      { x: 112, y: 122 },
      { x: 96, y: -30 },
    ],
  },
  {
    id: "orbit-capsule",
    variant: "capsule-blue",
    x: -168,
    y: -76,
    rotation: 28,
    scale: 0.76,
    delay: 2.7,
    floatDistance: 0,
    cycleDuration: 14.4,
    trajectory: [
      { x: 88, y: -28 },
      { x: 18, y: -132 },
      { x: -126, y: -154 },
      { x: -222, y: -32 },
      { x: -150, y: 92 },
      { x: 88, y: -28 },
    ],
  },
  {
    id: "orbit-oval",
    variant: "oval",
    x: 148,
    y: 108,
    rotation: -12,
    scale: 0.7,
    delay: 5.4,
    floatDistance: 0,
    cycleDuration: 13.6,
    trajectory: [
      { x: 100, y: -26 },
      { x: 188, y: -70 },
      { x: 232, y: 58 },
      { x: 152, y: 148 },
      { x: 18, y: 110 },
      { x: 100, y: -26 },
    ],
  },
  {
    id: "orbit-small-round",
    variant: "round",
    x: -122,
    y: 68,
    rotation: 14,
    scale: 0.58,
    delay: 8.1,
    floatDistance: 0,
    cycleDuration: 15.2,
    trajectory: [
      { x: 92, y: -24 },
      { x: 6, y: -76 },
      { x: -120, y: -42 },
      { x: -176, y: 72 },
      { x: -44, y: 132 },
      { x: 92, y: -24 },
    ],
  },
  {
    id: "orbit-warm-capsule",
    variant: "capsule",
    x: 224,
    y: 34,
    rotation: -32,
    scale: 0.64,
    delay: 10.8,
    floatDistance: 0,
    cycleDuration: 14.8,
    trajectory: [
      { x: 104, y: -34 },
      { x: 204, y: -118 },
      { x: 264, y: 4 },
      { x: 220, y: 116 },
      { x: 116, y: 144 },
      { x: 104, y: -34 },
    ],
  },
  {
    id: "orbit-high-oval",
    variant: "oval-blue",
    x: -12,
    y: -168,
    rotation: 42,
    scale: 0.62,
    delay: 13.5,
    floatDistance: 0,
    cycleDuration: 16.1,
    trajectory: [
      { x: 94, y: -32 },
      { x: 122, y: -158 },
      { x: 24, y: -208 },
      { x: -102, y: -146 },
      { x: -126, y: -24 },
      { x: 94, y: -32 },
    ],
  },
];

export function MedicineBox() {
  const reducedMotion = Boolean(useReducedMotion());
  const pointerX = useMotionValue(0);
  const pointerY = useMotionValue(0);
  const smoothX = useSpring(pointerX, { stiffness: 120, damping: 24, mass: 0.6 });
  const smoothY = useSpring(pointerY, { stiffness: 120, damping: 24, mass: 0.6 });
  const rotateY = useTransform(smoothX, [-1, 1], [-3.2, 3.2]);
  const rotateX = useTransform(smoothY, [-1, 1], [2.4, -2.4]);

  const moveScene = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (reducedMotion) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    pointerX.set(((event.clientX - bounds.left) / bounds.width) * 2 - 1);
    pointerY.set(((event.clientY - bounds.top) / bounds.height) * 2 - 1);
  };

  const resetScene = () => {
    pointerX.set(0);
    pointerY.set(0);
  };

  return (
    <motion.div
      className="medicine-stage"
      style={reducedMotion ? undefined : { rotateX, rotateY }}
      onPointerMove={moveScene}
      onPointerLeave={resetScene}
      role="img"
      aria-label="Şeffaf cam ilaç kutusu, cam şişe ve çevresinde yavaşça süzülen beyaz tabletler"
    >
      <motion.img
        src="/glass-box.png?v=3"
        alt=""
        className="glass-box-render"
        initial={false}
        animate={!reducedMotion ? { opacity: 1, y: [8, -8, 8] } : { opacity: 1, y: 0 }}
        transition={!reducedMotion ? { duration: 7, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" } : { duration: 0 }}
        aria-hidden="true"
      />

      {animatedPills.map((pill) => (
        <FloatingPill
          key={pill.id}
          {...pill}
          reducedMotion={reducedMotion}
          entranceReady
        />
      ))}

    </motion.div>
  );
}
