"use client";

import { motion } from "framer-motion";

const PARTICLES = Array.from({ length: 24 }, (_, index) => ({
  id: index,
  size: 2 + (index % 5) * 2,
  left: `${(index * 17) % 100}%`,
  delay: index * 0.25,
  duration: 8 + (index % 6),
}));

export function ParticlesBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 overflow-hidden">
      {PARTICLES.map((particle) => (
        <motion.span
          key={particle.id}
          className="absolute rounded-full bg-cyan/70 shadow-[0_0_18px_rgba(0,242,255,0.55)]"
          style={{
            width: particle.size,
            height: particle.size,
            left: particle.left,
            top: "110%",
          }}
          animate={{
            y: [0, -1200],
            opacity: [0, 0.8, 0],
            x: [0, particle.id % 2 === 0 ? 18 : -18, 0],
          }}
          transition={{
            duration: particle.duration,
            delay: particle.delay,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      ))}
    </div>
  );
}
