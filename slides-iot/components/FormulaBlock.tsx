"use client";

import { motion } from "framer-motion";

interface FormulaBlockProps {
  formula: string;
  description?: string;
  delay?: number;
  glow?: "cyan" | "magenta" | "violet";
}

export default function FormulaBlock({
  formula,
  description,
  delay = 0,
  glow = "cyan",
}: FormulaBlockProps) {
  const glowColors = {
    cyan: "border-neon-primary/30 text-glow-cyan",
    magenta: "border-neon-secondary/30 text-glow-magenta",
    violet: "border-neon-accent/30 text-glow-violet",
  };

  const textColors = {
    cyan: "text-neon-primary",
    magenta: "text-neon-secondary",
    violet: "text-neon-accent",
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, duration: 0.5 }}
      className={`p-6 rounded-xl bg-neon-bg/60 border ${glowColors[glow]} backdrop-blur-sm`}
    >
      <div className={`text-2xl font-mono text-center ${textColors[glow]} mb-2`}>
        {formula}
      </div>
      {description && (
        <p className="text-sm text-neon-muted text-center">{description}</p>
      )}
    </motion.div>
  );
}
