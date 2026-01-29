export interface SlideConfig {
  id: number;
  title: string;
  subtitle?: string;
  bgIntensity?: number;
  bgColor?: string;
}

export const SLIDES: SlideConfig[] = [
  { id: 1, title: "CIC-IIoT-2025", subtitle: "Security Analysis", bgIntensity: 150, bgColor: "#00f0ff" },
  { id: 2, title: "Agenda", subtitle: "Plan de présentation", bgIntensity: 60, bgColor: "#8b5cf6" },
  { id: 3, title: "Dataset", subtitle: "CIC-IIoT-2025", bgIntensity: 80, bgColor: "#00f0ff" },
  { id: 4, title: "Features", subtitle: "Caractéristiques discriminantes", bgIntensity: 70, bgColor: "#8b5cf6" },
  { id: 5, title: "Anomaly Detection", subtitle: "Méthodes non supervisées", bgIntensity: 70, bgColor: "#00f0ff" },
  { id: 6, title: "Résultats", subtitle: "Anomaly Detection", bgIntensity: 75, bgColor: "#ff00aa" },
  { id: 7, title: "Classification", subtitle: "Méthodes supervisées", bgIntensity: 70, bgColor: "#8b5cf6" },
  { id: 8, title: "Résultats", subtitle: "Classification", bgIntensity: 100, bgColor: "#00f0ff" },
  { id: 9, title: "FGSM Attack", subtitle: "Exploratory Attack", bgIntensity: 90, bgColor: "#ff00aa" },
  { id: 10, title: "Causative Attack", subtitle: "Data Poisoning", bgIntensity: 70, bgColor: "#ff00aa" },
  { id: 11, title: "Robustesse", subtitle: "Comparaison des modèles", bgIntensity: 50, bgColor: "#8b5cf6" },
  { id: 12, title: "Summary", subtitle: "Meilleurs modèles", bgIntensity: 60, bgColor: "#00f0ff" },
  { id: 13, title: "Recommandations", subtitle: "Défense multi-couches", bgIntensity: 70, bgColor: "#8b5cf6" },
  { id: 14, title: "Questions", bgIntensity: 200, bgColor: "#ff00aa" },
];

export const TOTAL_SLIDES = SLIDES.length;

export const TEAM_MEMBERS = [
  "Alexis Le Trung",
  "Yahya Ahachim",
  "Rayan Drissi",
  "Aniss Outaleb",
];
