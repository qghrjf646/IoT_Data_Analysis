"use client";

import { motion } from "framer-motion";
import NeonCard from "@/components/NeonCard";
import AnimatedCounter from "@/components/AnimatedCounter";
import Image from "next/image";

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
        className="relative z-10 text-7xl font-display font-bold text-neon-primary mb-2"
      >
        CIC-IIoT-2025 Dataset
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-xl text-neon-muted mb-8"
      >
        Industrial IoT Network Traffic Dataset
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-2 gap-8">
        <div className="flex flex-col gap-6">
          <NeonCard delay={0.4} glow="cyan" title="Statistics">
            <div className="space-y-4 mt-4">
              {stats.map((stat, i) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 + i * 0.1 }}
                  className="flex justify-between items-center py-2 border-b border-neon-muted/20"
                >
                  <span className="text-neon-text">{stat.label}</span>
                  <span className="text-neon-primary font-mono font-bold">
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
            className="p-4 bg-neon-bg/60 border border-neon-accent/30 rounded-xl"
          >
            <h4 className="text-neon-accent font-bold mb-3">Attack Categories:</h4>
            <div className="flex flex-wrap gap-2">
              {categories.map((cat, i) => (
                <motion.span
                  key={cat}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 1.4 + i * 0.08 }}
                  className="px-3 py-1 bg-neon-secondary/10 border border-neon-secondary/30 rounded-full text-sm text-neon-secondary"
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
          className="rounded-xl overflow-hidden border-2 border-neon-primary/30 bg-neon-bg/40 flex items-center justify-center p-4"
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
