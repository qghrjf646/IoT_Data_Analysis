"use client";

import { motion } from "framer-motion";
import NeonCard from "@/components/NeonCard";
import { Search, Shield, Swords } from "lucide-react";

export default function Slide12Summary() {
  const tasks = [
    {
      task: "Zero-day Detection",
      model: "Local Outlier Factor",
      metric: "F1 = 0.831, AUPRC = 0.873",
      icon: Search,
      color: "cyan" as const,
    },
    {
      task: "Attack Classification",
      model: "Random Forest",
      metric: "F1 = 0.927, AUPRC = 0.946",
      icon: Shield,
      color: "violet" as const,
    },
    {
      task: "Adversarial Robustness",
      model: "Random Forest",
      metric: "44.2% robust acc.",
      icon: Swords,
      color: "magenta" as const,
    },
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-5xl md:text-7xl font-display font-bold text-neon-primary mb-2 leading-tight"
      >
        Summary: Best Models by Task
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-xl text-neon-muted mb-12"
      >
        Récapitulatif des meilleurs modèles
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {tasks.map((t, i) => (
          <NeonCard
            key={t.task}
            delay={0.5 + i * 0.2}
            glow={t.color}
            className="flex flex-col items-center text-center"
          >
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.8 + i * 0.2, type: "spring" }}
              className="mb-6"
            >
              <t.icon
                size={64}
                strokeWidth={1.5}
                className={`
                  ${t.color === "cyan" ? "text-neon-primary" : ""}
                  ${t.color === "magenta" ? "text-neon-secondary" : ""}
                  ${t.color === "violet" ? "text-neon-accent" : ""}
                `}
              />
            </motion.div>
            <h3 className="text-xl font-display font-bold text-neon-text mb-4">{t.task}</h3>
            <div className={`
              text-2xl font-bold mb-4
              ${t.color === "cyan" ? "text-neon-primary" : ""}
              ${t.color === "magenta" ? "text-neon-secondary" : ""}
              ${t.color === "violet" ? "text-neon-accent" : ""}
            `}>
              {t.model}
            </div>
            <div className="p-3 bg-neon-bg/60 border border-neon-muted/30 rounded-lg">
              <span className="font-mono text-sm text-neon-muted">{t.metric}</span>
            </div>
          </NeonCard>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.4 }}
        className="relative z-10 mt-8 text-center"
      >
        <div className="inline-block p-6 bg-neon-secondary/10 border border-neon-secondary/30 rounded-xl">
          <p className="text-xl text-neon-text">
            <span className="text-neon-secondary font-bold">Key Insight:</span>{" "}
            No single model excels at all tasks — <span className="text-neon-primary font-bold">defense-in-depth</span> required
          </p>
        </div>
      </motion.div>
    </div>
  );
}
