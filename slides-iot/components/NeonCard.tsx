"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

interface NeonCardProps {
  children: ReactNode;
  className?: string;
  delay?: number;
  glow?: "cyan" | "magenta" | "violet";
  title?: string;
}

export default function NeonCard({
  children,
  className = "",
  delay = 0,
  glow = "cyan",
  title,
}: NeonCardProps) {
  const glowColors = {
    cyan: "border-neon-primary/50 shadow-neon-cyan",
    magenta: "border-neon-secondary/50 shadow-neon-magenta",
    violet: "border-neon-accent/50 shadow-neon-violet",
  };

  const textColors = {
    cyan: "text-neon-primary",
    magenta: "text-neon-secondary",
    violet: "text-neon-accent",
  };

  const beamColors = {
    cyan: "from-transparent via-neon-primary to-transparent",
    magenta: "from-transparent via-neon-secondary to-transparent",
    violet: "from-transparent via-neon-accent to-transparent",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: "easeOut" }}
      className={`
        bg-neon-bg/80 backdrop-blur-md rounded-xl border-2 p-8 relative overflow-hidden group
        ${glowColors[glow]}
        ${className}
      `}
    >
      <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-xl">
        <motion.div
          animate={{
            top: ["-100%", "100%", "100%", "-100%", "-100%"],
            left: ["-100%", "-100%", "100%", "100%", "-100%"],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "linear",
          }}
          className={`absolute w-32 h-32 bg-gradient-to-r ${beamColors[glow]} blur-xl opacity-40`}
        />
      </div>

      {title && (
        <div className="absolute -top-4 left-6 bg-neon-bg px-3 z-10">
          <h3 className={`text-lg font-bold tracking-wider uppercase ${textColors[glow]}`}>
            {title}
          </h3>
        </div>
      )}
      <div className="h-full">
        {children}
      </div>
    </motion.div>
  );
}
