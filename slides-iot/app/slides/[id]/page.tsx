import SlideContainer from "@/components/SlideContainer";
import Slide01Title from "@/lib/slides/01-title";
import Slide02Agenda from "@/lib/slides/02-agenda";
import Slide03Dataset from "@/lib/slides/03-dataset";
import Slide04Features from "@/lib/slides/04-features";
import Slide05AnomalyMethods from "@/lib/slides/05-anomaly-methods";
import Slide06AnomalyResults from "@/lib/slides/06-anomaly-results";
import Slide07ClassificationMethods from "@/lib/slides/07-classification-methods";
import Slide08ClassificationResults from "@/lib/slides/08-classification-results";
import Slide09FGSMAttack from "@/lib/slides/09-fgsm-attack";
import Slide10CausativeAttack from "@/lib/slides/10-causative-attack";
import Slide11Robustness from "@/lib/slides/11-robustness";
import Slide12Summary from "@/lib/slides/12-summary";
import Slide13Recommendations from "@/lib/slides/13-recommendations";
import Slide14Questions from "@/lib/slides/14-questions";
import { TOTAL_SLIDES } from "@/lib/slides-config";

const SLIDE_COMPONENTS: Record<number, React.ComponentType> = {
  1: Slide01Title,
  2: Slide02Agenda,
  3: Slide03Dataset,
  4: Slide04Features,
  5: Slide05AnomalyMethods,
  6: Slide06AnomalyResults,
  7: Slide07ClassificationMethods,
  8: Slide08ClassificationResults,
  9: Slide09FGSMAttack,
  10: Slide10CausativeAttack,
  11: Slide11Robustness,
  12: Slide12Summary,
  13: Slide13Recommendations,
  14: Slide14Questions,
};

export function generateStaticParams() {
  return Array.from({ length: TOTAL_SLIDES }, (_, i) => ({
    id: String(i + 1),
  }));
}

interface PageProps {
  params: Promise<{ id: string }>;
}

export default async function SlidePage({ params }: PageProps) {
  const { id } = await params;
  const slideId = parseInt(id, 10);

  const SlideComponent = SLIDE_COMPONENTS[slideId];

  if (!SlideComponent) {
    return (
      <SlideContainer slideNumber={1}>
        <div className="flex-1 flex items-center justify-center">
          <p className="text-neon-secondary text-2xl">Slide non trouvée</p>
        </div>
      </SlideContainer>
    );
  }

  return (
    <SlideContainer slideNumber={slideId}>
      <SlideComponent />
    </SlideContainer>
  );
}
