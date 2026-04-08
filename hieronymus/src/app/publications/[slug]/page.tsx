import { notFound } from 'next/navigation';
import Link from 'next/link';
import { papers, getPaperBySlug } from '@/data/papers';
import PaperDetail from './PaperDetail';

export function generateStaticParams() {
  return papers.map((p) => ({ slug: p.slug }));
}

export function generateMetadata({ params }: { params: { slug: string } }) {
  const paper = getPaperBySlug(params.slug);
  if (!paper) return { title: 'Paper not found' };
  return {
    title: `${paper.title} - Hieronymus`,
    description: paper.subtitle,
  };
}

export default function PaperPage({ params }: { params: { slug: string } }) {
  const paper = getPaperBySlug(params.slug);
  if (!paper) notFound();

  return <PaperDetail paper={paper} />;
}
