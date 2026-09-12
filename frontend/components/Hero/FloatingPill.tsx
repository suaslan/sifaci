"use client";

import { motion } from "motion/react";

export type PillVariant =
  | "round"
  | "oval"
  | "oval-blue"
  | "capsule"
  | "capsule-blue"
  | "capsule-navy";

export type FloatingPillProps = {
  id: string;
  variant: PillVariant;
  x: number;
  y: number;
  rotation: number;
  scale: number;
  delay: number;
  floatDistance: number;
  trajectory?: Array<{ x: number; y: number }>;
  cycleDuration?: number;
  reducedMotion: boolean;
  entranceReady: boolean;
};

function PillBody({ variant }: Pick<FloatingPillProps, "variant">) {
  if (variant.startsWith("capsule")) {
    const toneClass =
      variant === "capsule-blue"
        ? "pill-capsule--blue"
        : variant === "capsule-navy"
          ? "pill-capsule--navy"
          : "";

    return (
      <span className={`pill-shell pill-capsule ${toneClass}`}>
        <span className="pill-capsule__half" />
        <span className="pill-capsule__half" />
      </span>
    );
  }

  if (variant.startsWith("oval")) {
    return (
      <span
        className={`pill-shell pill-oval ${variant === "oval-blue" ? "pill-oval--blue" : ""}`}
      />
    );
  }

  return <span className="pill-shell pill-round" />;
}

export function FloatingPill({
  id,
  variant,
  x,
  y,
  rotation,
  scale,
  delay,
  floatDistance,
  trajectory,
  cycleDuration = 12.5,
  reducedMotion,
  entranceReady,
}: FloatingPillProps) {
  return (
    <motion.div
      className="floating-pill"
      initial={false}
      animate={
        trajectory && entranceReady && !reducedMotion
          ? {
              opacity: [0, 1, 1, 1, 1, 0],
              x: trajectory.map((point) => point.x),
              y: trajectory.map((point) => point.y),
              rotateZ: [rotation * 0.2, rotation, rotation + 18, rotation - 12, rotation, rotation],
              scale: [0.34, scale, scale * 0.95, scale * 0.9, scale * 0.85, 0.34],
            }
          : entranceReady && !reducedMotion
            ? {
                opacity: [0, 1],
                x: [0, x],
                y: [56, y],
                rotateZ: [rotation * 0.12, rotation],
                scale: [0.42, scale],
              }
            : { opacity: 1, x, y, rotateZ: rotation, scale }
      }
      transition={
        reducedMotion
          ? { duration: 0 }
          : trajectory
            ? {
                duration: cycleDuration,
                delay: 1.2 + delay,
                repeat: Number.POSITIVE_INFINITY,
                repeatDelay: 2.4,
                ease: [0.22, 1, 0.36, 1],
              }
          : {
              type: "spring",
              stiffness: 74,
              damping: 16,
              mass: 0.94,
              delay: 0.96 + delay,
            }
      }
      data-pill={id}
      aria-hidden="true"
    >
      <motion.span
        className="block"
        animate={
          reducedMotion
            ? undefined
            : {
                y: [0, -floatDistance, 0],
                rotateZ: [0, id.length % 2 === 0 ? 3 : -3, 0],
              }
        }
        transition={{
          duration: 4.9 + delay,
          delay: 2.35 + delay,
          ease: "easeInOut",
          repeat: Number.POSITIVE_INFINITY,
        }}
      >
        <PillBody variant={variant} />
      </motion.span>
    </motion.div>
  );
}
