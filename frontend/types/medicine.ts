export type MedicineAnswer = {
  answer: string;
  medicine?: string | null;
  sources: string[];
  disclaimer: string;
  suggestions?: string[];
};
