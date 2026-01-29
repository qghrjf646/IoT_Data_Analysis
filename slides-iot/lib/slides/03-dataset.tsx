"use client";

import { motion } from "framer-motion";
import NeonCard from "@/components/NeonCard";
import AnimatedCounter from "@/components/AnimatedCounter";

export default function Slide03Dataset() {
  const stats = [
    { label: "Total Samples", value: 227191, suffix: "" },
    { label: "Features", value: 94, suffix: "" },
    { label: "Attack Samples", value: 90391, suffix: " (39.8%)" },
    { label: "Benign Samples", value: 136800, suffix: " (60.2%)" },
    { label: "Attack Categories", value: 7, suffix: "" },
  ];

  const categories = [
    "Reconnaissance",
    "DoS",
    "DDoS", 
    "MitM",
    "Malware",
    "Web",
    "Brute Force"
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-5xl md:text-6xl font-display font-bold text-neon-primary mb-4 mt-4 leading-tight"
      >
        CIC-IIoT-2025 Dataset
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-lg text-neon-muted mb-6"
      >
        Industrial IoT Network Traffic Dataset
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-2 gap-6 min-h-0">
        <div className="flex flex-col gap-4 min-h-0 overflow-hidden">
          <NeonCard delay={0.4} glow="cyan" title="Statistics" className="flex-1 min-h-0 overflow-auto">
            <div className="space-y-3 mt-4">
              {stats.map((stat, i) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 + i * 0.1 }}
                  className="flex justify-between items-center py-1.5 border-b border-neon-muted/20"
                >
                  <span className="text-neon-text text-sm">{stat.label}</span>
                  <span className="text-neon-primary font-mono font-bold text-sm">
                    <AnimatedCounter value={stat.value} delay={0.8 + i * 0.1} duration={1.5} />
                    {stat.suffix}
                  </span>
                </motion.div>
              ))}
            </div>
          </NeonCard>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2 }}
            className="p-4 bg-neon-bg/60 border border-neon-accent/30 rounded-xl shrink-0"
          >
            <h4 className="text-neon-accent font-bold mb-2 text-sm">Attack Categories:</h4>
            <div className="flex flex-wrap gap-1.5">
              {categories.map((cat, i) => (
                <motion.span
                  key={cat}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1.4 + i * 0.08 }}
                  className="px-2 py-0.5 bg-neon-secondary/10 border border-neon-secondary/30 rounded-full text-xs text-neon-secondary"
                >
                  {cat}
                </motion.span>
              ))}
            </div>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.8 }}
          className="rounded-xl overflow-hidden border-2 border-neon-primary/30 bg-neon-bg/40 flex items-center justify-center p-4 min-h-0"
        >
          <img
            src="/figures/class_distribution.png"
            alt="Class Distribution"
            className="max-w-full max-h-full object-contain"
          />
        </motion.div>
      </div>
    </div>
  );
}
