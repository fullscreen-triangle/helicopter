import { Montserrat } from 'next/font/google';
import './globals.css';
import NavbarApp from '@/components/NavbarApp';
import FooterApp from '@/components/FooterApp';

const montserrat = Montserrat({
  subsets: ['latin'],
  variable: '--font-mont',
  display: 'swap',
});

export const metadata = {
  title: 'Hieronymus - Universal Observation Platform',
  description: 'Client-side GPU observation engine for microscopy, spectroscopy, and multi-modal scientific imaging.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${montserrat.variable} font-mont bg-dark text-light min-h-screen flex flex-col`}
      >
        <NavbarApp />
        <main className="flex-1">{children}</main>
        <FooterApp />
      </body>
    </html>
  );
}
