"use client";

import { motion } from "framer-motion";

interface AnimatedBarProps {
  value: number;
  maxValue?: number;
  label: string;
  color?: "cyan" | "magenta" | "violet";
  delay?: number;
}

export default function AnimatedBar({
  value,
  maxValue = 1,
  label,
  color = "cyan",
  delay = 0,
}: AnimatedBarProps) {
  const percentage = (value / maxValue) * 100;

  const colors = {
    cyan: "bg-neon-primary",
    magenta: "bg-neon-secondary",
    violet: "bg-neon-accent",
  };

  const glowColors = {
    cyan: "shadow-neon-cyan",
    magenta: "shadow-neon-magenta",
    violet: "shadow-neon-violet",
  };

  return (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-neon-text">{label}</span>
        <span className="text-neon-muted font-mono">{value.toFixed(3)}</span>
      </div>
      <div className="h-3 bg-neon-bg/50 rounded-full overflow-hidden border border-neon-muted/20">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ delay, duration: 1, ease: "easeOut" }}
          className={`h-full ${colors[color]} ${glowColors[color]} rounded-full`}
        />
      </div>
    </div>
  );
}
