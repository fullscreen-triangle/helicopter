#!/usr/bin/env python3
"""Remove all internal cross-references from LaTeX file while preserving citations."""

import re
import sys

def remove_internal_references(text):
    """Remove all internal cross-references from LaTeX text."""
    
    # Remove custom reference commands from preamble
    text = re.sub(r'\\newcommand\{\\figref\}\[1\]\{Figure~\\ref\{#1\}\}\n?', '', text)
    text = re.sub(r'\\newcommand\{\\eqnref\}\[1\]\{Equation~\\ref\{#1\}\}\n?', '', text)
    text = re.sub(r'\\newcommand\{\\secref\}\[1\]\{Section~\\ref\{#1\}\}\n?', '', text)
    
    # Pattern matching for different reference types with context
    
    # Definition~\ref{def:*} -> "the definition of X" or just remove
    text = re.sub(r'Definition~\\ref\{def:[^}]+\}', 'the definition', text)
    
    # Theorem~\ref{thm:*} -> "the theorem" or just remove
    text = re.sub(r'Theorem~\\ref\{thm:[^}]+\}', 'the theorem', text)
    
    # Axiom~\ref{ax:*} -> "the axiom"
    text = re.sub(r'Axiom~\\ref\{ax:[^}]+\}', 'the axiom', text)
    
    # Corollary~\ref{cor:*} -> "the corollary"
    text = re.sub(r'Corollary~\\ref\{cor:[^}]+\}', 'the corollary', text)
    
    # Lemma~\ref{lem:*} -> "the lemma"
    text = re.sub(r'Lemma~\\ref\{lem:[^}]+\}', 'the lemma', text)
    
    # Section~\ref{sec:*} -> "an earlier section"
    text = re.sub(r'Section~\\ref\{sec:[^}]+\}(?:\.\\ref\{subsec:[^}]+\})?', 'an earlier section', text)
    
    # Equation~\ref{eq:*} or \eqref{eq:*} -> just remove the reference
    text = re.sub(r'Equation~\\ref\{eq:[^}]+\}', 'the equation', text)
    text = re.sub(r'\\eqref\{eq:[^}]+\}', '', text)
    
    # Figure~\ref{fig:*} or \figref{fig:*} -> "the figure" or just remove
    text = re.sub(r'Figure~\\ref\{fig:[^}]+\}', 'the figure', text)
    text = re.sub(r'\\figref\{fig:[^}]+\}', 'the figure', text)
    
    # Table~\ref{tab:*} -> "the table"
    text = re.sub(r'Table~\\ref\{tab:[^}]+\}', 'the table', text)
    
    # Parenthetical references like (Theorem~\ref{thm:*}) -> remove entirely
    text = re.sub(r'\(Definition~\\ref\{def:[^}]+\}\)', '', text)
    text = re.sub(r'\(Theorem~\\ref\{thm:[^}]+\}\)', '', text)
    text = re.sub(r'\(Axiom~\\ref\{ax:[^}]+\}\)', '', text)
    text = re.sub(r'\(Corollary~\\ref\{cor:[^}]+\}\)', '', text)
    text = re.sub(r'\(Lemma~\\ref\{lem:[^}]+\}\)', '', text)
    text = re.sub(r'\(Section~\\ref\{sec:[^}]+\}\)', '', text)
    
    # Multiple references in parentheses like (Theorems~\ref{}, \ref{})
    text = re.sub(r'\(Theorems~\\ref\{[^}]+\},?\s*\\ref\{[^}]+\}\)', '', text)
    
    # Generic \ref{anything} not already caught -> remove
    text = re.sub(r'\\ref\{[^}]+\}', '', text)
    text = re.sub(r'\\Cref\{[^}]+\}', 'the earlier result', text)
    text = re.sub(r'\\cref\{[^}]+\}', 'the earlier result', text)
    
    # Clean up double spaces and spaces before punctuation
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r' +([.,;:])', r'\1', text)
    text = re.sub(r'\( \)', '', text)
    text = re.sub(r'\(\)', '', text)
    
    # Clean up "as established in  ," or similar
    text = re.sub(r'as established in\s+,', 'as established,', text)
    text = re.sub(r'as derived in\s+,', 'as derived,', text)
    text = re.sub(r'as shown in\s+,', 'as shown,', text)
    
    return text

def main():
    input_file = 'pixel_maxwell_demon/docs/lunar-surface/lunar-surface-arxiv-submission.tex'
    output_file = 'pixel_maxwell_demon/docs/lunar-surface/lunar-surface-arxiv-submission-fixed.tex'
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Removing internal references...")
    content = remove_internal_references(content)
    
    print(f"Writing {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Done!")
    print(f"\nOutput written to: {output_file}")

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
