# Diagrams 

# Dualmode Overview 
`<svg xmlns="http://www.w3.org/2000/svg" width="820" height="260">
<defs>
<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
<polygon points="0,0 10,5 0,10" fill="black"/>
</marker>
</defs>
<rect x="20" y="100" width="100" height="50" fill="none" stroke="black"/>
<text x="70" y="130" font-size="12" text-anchor="middle">Multi-Modal Input</text>

<rect x="160" y="100" width="120" height="50" fill="none" stroke="black"/> <text x="220" y="118" font-size="12" text-anchor="middle">Mode Selector</text> <text x="220" y="135" font-size="10" text-anchor="middle">complexity + intent</text> <rect x="330" y="40" width="170" height="70" fill="none" stroke="black"/> <text x="415" y="60" font-size="12" text-anchor="middle">Assistant Mode</text> <text x="415" y="78" font-size="10" text-anchor="middle">Step-by-step CV</text> <text x="415" y="92" font-size="10" text-anchor="middle">Variance tracked</text> <rect x="330" y="150" width="170" height="70" fill="none" stroke="black"/> <text x="415" y="170" font-size="12" text-anchor="middle">Turbulence Mode</text> <text x="415" y="188" font-size="10" text-anchor="middle">BMD cross-product</text> <text x="415" y="202" font-size="10" text-anchor="middle">Direct equilibrium</text> <rect x="540" y="95" width="140" height="60" fill="none" stroke="black"/> <text x="610" y="115" font-size="12" text-anchor="middle">Variance</text> <text x="610" y="130" font-size="12" text-anchor="middle">Minimized Output</text> <rect x="700" y="95" width="100" height="60" fill="none" stroke="black"/> <text x="750" y="120" font-size="12" text-anchor="middle">User / System</text> <text x="750" y="136" font-size="10" text-anchor="middle">Consumption</text> <line x1="120" y1="125" x2="160" y2="125" stroke="black" marker-end="url(#arrow)"/> <line x1="280" y1="125" x2="330" y2="75" stroke="black" marker-end="url(#arrow)"/> <line x1="280" y1="125" x2="330" y2="195" stroke="black" marker-end="url(#arrow)"/> <line x1="500" y1="75" x2="540" y2="125" stroke="black" marker-end="url(#arrow)"/> <line x1="500" y1="185" x2="540" y2="125" stroke="black" marker-end="url(#arrow)"/> <line x1="680" y1="125" x2="700" y2="125" stroke="black" marker-end="url(#arrow)"/> </svg>`
Description : Dual-Mode Overview (Input → Mode Selector → Two Pipelines → Unified Output)
Alt text: Diagram showing input analyzed by a mode selector that routes to Assistant Mode or Turbulence Mode, both feeding a variance-minimized output.

#  Assistant Mode Pipeline with Variance Tracking
`<svg xmlns="http://www.w3.org/2000/svg" width="880" height="230">
<defs>
<marker id="arrow2" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
<polygon points="0,0 10,5 0,10" fill="black"/>
</marker>
</defs>
<rect x="20" y="90" width="120" height="50" fill="none" stroke="black"/>
<text x="80" y="110" font-size="11" text-anchor="middle">Thermodynamic</text>
<text x="80" y="125" font-size="11" text-anchor="middle">Pixel Proc</text>

<rect x="170" y="90" width="120" height="50" fill="none" stroke="black"/> <text x="230" y="110" font-size="11" text-anchor="middle">Hierarchical</text> <text x="230" y="125" font-size="11" text-anchor="middle">Bayesian</text> <rect x="320" y="90" width="140" height="50" fill="none" stroke="black"/> <text x="390" y="110" font-size="11" text-anchor="middle">Autonomous</text> <text x="390" y="125" font-size="11" text-anchor="middle">Reconstruction</text> <rect x="490" y="90" width="140" height="50" fill="none" stroke="black"/> <text x="560" y="110" font-size="11" text-anchor="middle">Variance</text> <text x="560" y="125" font-size="11" text-anchor="middle">Validation</text> <rect x="660" y="90" width="180" height="50" fill="none" stroke="black"/> <text x="750" y="110" font-size="11" text-anchor="middle">Human-Compatible</text> <text x="750" y="125" font-size="11" text-anchor="middle">Explanation Output</text> <line x1="140" y1="115" x2="170" y2="115" stroke="black" marker-end="url(#arrow2)"/> <line x1="290" y1="115" x2="320" y2="115" stroke="black" marker-end="url(#arrow2)"/> <line x1="460" y1="115" x2="490" y2="115" stroke="black" marker-end="url(#arrow2)"/> <line x1="630" y1="115" x2="660" y2="115" stroke="black" marker-end="url(#arrow2)"/> <rect x="100" y="20" width="560" height="30" fill="none" stroke="black" stroke-dasharray="4 4"/> <text x="380" y="40" font-size="11" text-anchor="middle">Variance Tracker (Deviation from Equilibrium Accumulated & Propagated)</text> <line x1="80" y1="90" x2="80" y2="50" stroke="black"/> <line x1="230" y1="90" x2="230" y2="50" stroke="black"/> <line x1="390" y1="90" x2="390" y2="50" stroke="black"/> <line x1="560" y1="90" x2="560" y2="50" stroke="black"/> <rect x="20" y="170" width="140" height="40" fill="none" stroke="black"/> <text x="90" y="188" font-size="11" text-anchor="middle">User Feedback</text> <text x="90" y="202" font-size="10" text-anchor="middle">Adjust variance</text> <line x1="90" y1="170" x2="80" y2="140" stroke="black" marker-end="url(#arrow2)"/> <line x1="90" y1="170" x2="230" y2="140" stroke="black" marker-end="url(#arrow2)"/> <line x1="90" y1="170" x2="390" y2="140" stroke="black" marker-end="url(#arrow2)"/> <line x1="90" y1="170" x2="560" y2="140" stroke="black" marker-end="url(#arrow2)"/> </svg>`
Description: Linear pipeline: thermodynamic pixel processing, hierarchical Bayesian, reconstruction, explanation; variance tracker spans all stages.

# Diagram 3. Turbulence Mode Pipeline
Alt text: Pipeline showing BMD extraction, cross-product, S-entropy navigation, equilibrium navigation, consciousness validation leading to solution.
`<svg xmlns="http://www.w3.org/2000/svg" width="900" height="190">
<defs>
<marker id="arrow3" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
<polygon points="0,0 10,5 0,10" fill="black"/>
</marker>
</defs>
<rect x="20" y="70" width="110" height="50" fill="none" stroke="black"/>
<text x="75" y="90" font-size="11" text-anchor="middle">BMD</text>
<text x="75" y="105" font-size="11" text-anchor="middle">Extraction</text>

<rect x="150" y="70" width="150" height="50" fill="none" stroke="black"/> <text x="225" y="90" font-size="11" text-anchor="middle">Cross-Product</text> <text x="225" y="105" font-size="11" text-anchor="middle">Constraint Manifold</text> <rect x="320" y="70" width="150" height="50" fill="none" stroke="black"/> <text x="395" y="90" font-size="11" text-anchor="middle">S-Entropy</text> <text x="395" y="105" font-size="11" text-anchor="middle">Navigation</text> <rect x="490" y="70" width="150" height="50" fill="none" stroke="black"/> <text x="565" y="90" font-size="11" text-anchor="middle">Direct Equilibrium</text> <text x="565" y="105" font-size="11" text-anchor="middle">Navigation</text> <rect x="660" y="70" width="110" height="50" fill="none" stroke="black"/> <text x="715" y="90" font-size="11" text-anchor="middle">Consciousness</text> <text x="715" y="105" font-size="11" text-anchor="middle">Validation</text> <rect x="790" y="70" width="90" height="50" fill="none" stroke="black"/> <text x="835" y="95" font-size="11" text-anchor="middle">Solution</text> <line x1="130" y1="95" x2="150" y2="95" stroke="black" marker-end="url(#arrow3)"/> <line x1="300" y1="95" x2="320" y2="95" stroke="black" marker-end="url(#arrow3)"/> <line x1="470" y1="95" x2="490" y2="95" stroke="black" marker-end="url(#arrow3)"/> <line x1="640" y1="95" x2="660" y2="95" stroke="black" marker-end="url(#arrow3)"/> <line x1="770" y1="95" x2="790" y2="95" stroke="black" marker-end="url(#arrow3)"/> <rect x="200" y="10" width="300" height="30" fill="none" stroke="black" stroke-dasharray="4 4"/> <text x="350" y="30" font-size="11" text-anchor="middle">Minimal Variance Emergence (Not iterative computation)</text> <rect x="540" y="140" width="180" height="40" fill="none" stroke="black"/> <text x="630" y="160" font-size="11" text-anchor="middle">Variance Gradient = 0 at Equilibrium</text> <line x1="630" y1="140" x2="630" y2="120" stroke="black" marker-end="url(#arrow3)"/> </svg>`



Diagram 4. BMD Cross-Product Variance Engine
Alt text: Three BMD modality inputs converted to gas molecules, tensor product forming constraint manifold, equilibrium surface extracted, variance path minimized.
`<svg xmlns="http://www.w3.org/2000/svg" width="880" height="300">
<defs>
<marker id="arrow4" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
<polygon points="0,0 10,5 0,10" fill="black"/>
</marker>
</defs>
<rect x="40" y="40" width="110" height="50" fill="none" stroke="black"/>
<text x="95" y="60" font-size="11" text-anchor="middle">Visual BMDs</text>
<text x="95" y="75" font-size="10" text-anchor="middle">→ molecules</text>

<rect x="40" y="110" width="110" height="50" fill="none" stroke="black"/> <text x="95" y="130" font-size="11" text-anchor="middle">Audio BMDs</text> <text x="95" y="145" font-size="10" text-anchor="middle">→ molecules</text> <rect x="40" y="180" width="110" height="50" fill="none" stroke="black"/> <text x="95" y="200" font-size="11" text-anchor="middle">Semantic BMDs</text> <text x="95" y="215" font-size="10" text-anchor="middle">→ molecules</text> <rect x="200" y="90" width="150" height="110" fill="none" stroke="black"/> <text x="275" y="115" font-size="11" text-anchor="middle">Tensor / Cross</text> <text x="275" y="130" font-size="11" text-anchor="middle">Product</text> <text x="275" y="150" font-size="10" text-anchor="middle">Constraint Manifold</text> <rect x="390" y="90" width="140" height="110" fill="none" stroke="black"/> <text x="460" y="120" font-size="11" text-anchor="middle">Equilibrium</text> <text x="460" y="135" font-size="11" text-anchor="middle">Surface Finder</text> <rect x="560" y="40" width="140" height="70" fill="none" stroke="black"/> <text x="630" y="65" font-size="11" text-anchor="middle">Variance</text> <text x="630" y="80" font-size="11" text-anchor="middle">Gradient</text> <rect x="560" y="150" width="140" height="70" fill="none" stroke="black"/> <text x="630" y="175" font-size="11" text-anchor="middle">Minimization</text> <text x="630" y="190" font-size="11" text-anchor="middle">Path</text> <rect x="740" y="110" width="110" height="70" fill="none" stroke="black"/> <text x="795" y="140" font-size="11" text-anchor="middle">Minimal</text> <text x="795" y="155" font-size="11" text-anchor="middle">Variance State</text> <line x1="150" y1="65" x2="200" y2="120" stroke="black" marker-end="url(#arrow4)"/> <line x1="150" y1="135" x2="200" y2="135" stroke="black" marker-end="url(#arrow4)"/> <line x1="150" y1="205" x2="200" y2="150" stroke="black" marker-end="url(#arrow4)"/> <line x1="350" y1="145" x2="390" y2="145" stroke="black" marker-end="url(#arrow4)"/> <line x1="530" y1="145" x2="560" y2="75" stroke="black" marker-end="url(#arrow4)"/> <line x1="530" y1="145" x2="560" y2="185" stroke="black" marker-end="url(#arrow4)"/> <line x1="700" y1="75" x2="740" y2="145" stroke="black" marker-end="url(#arrow4)"/> <line x1="700" y1="185" x2="740" y2="145" stroke="black" marker-end="url(#arrow4)"/> <rect x="200" y="10" width="330" height="20" fill="none" stroke="black" stroke-dasharray="4 4"/> <text x="365" y="25" font-size="10" text-anchor="middle">Conversion: BMD → Gas Molecular Representation</text> </svg>`

# Diagram 5. Mode Selection Logic
Alt text: Inputs to complexity assessment, consciousness markers, user preference feeding a decision node selecting assistant, turbulence, or hybrid.
`<svg xmlns="http://www.w3.org/2000/svg" width="780" height="260">
<defs>
<marker id="arrow5" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
<polygon points="0,0 10,5 0,10" fill="black"/>
</marker>
</defs>
<rect x="40" y="30" width="140" height="50" fill="none" stroke="black"/>
<text x="110" y="55" font-size="11" text-anchor="middle">Input Data</text>

<rect x="40" y="110" width="140" height="50" fill="none" stroke="black"/> <text x="110" y="130" font-size="11" text-anchor="middle">User Intent</text> <rect x="40" y="190" width="140" height="50" fill="none" stroke="black"/> <text x="110" y="210" font-size="11" text-anchor="middle">Context</text> <rect x="230" y="30" width="150" height="50" fill="none" stroke="black"/> <text x="305" y="50" font-size="11" text-anchor="middle">Complexity Score</text> <rect x="230" y="110" width="150" height="50" fill="none" stroke="black"/> <text x="305" y="130" font-size="11" text-anchor="middle">Consciousness</text> <text x="305" y="145" font-size="10" text-anchor="middle">Indicators</text> <rect x="230" y="190" width="150" height="50" fill="none" stroke="black"/> <text x="305" y="210" font-size="11" text-anchor="middle">Preference</text> <rect x="420" y="100" width="120" height="70" fill="none" stroke="black"/> <text x="480" y="125" font-size="11" text-anchor="middle">Decision</text> <text x="480" y="140" font-size="10" text-anchor="middle">Mode Selector</text> <rect x="580" y="30" width="150" height="50" fill="none" stroke="black"/> <text x="655" y="55" font-size="11" text-anchor="middle">Assistant Mode</text> <rect x="580" y="110" width="150" height="50" fill="none" stroke="black"/> <text x="655" y="135" font-size="11" text-anchor="middle">Turbulence Mode</text> <rect x="580" y="190" width="150" height="50" fill="none" stroke="black"/> <text x="655" y="215" font-size="11" text-anchor="middle">Hybrid Mode</text> <line x1="180" y1="55" x2="230" y2="55" stroke="black" marker-end="url(#arrow5)"/> <line x1="180" y1="135" x2="230" y2="135" stroke="black" marker-end="url(#arrow5)"/> <line x1="180" y1="215" x2="230" y2="215" stroke="black" marker-end="url(#arrow5)"/> <line x1="380" y1="55" x2="420" y2="135" stroke="black" marker-end="url(#arrow5)"/> <line x1="380" y1="135" x2="420" y2="135" stroke="black" marker-end="url(#arrow5)"/> <line x1="380" y1="215" x2="420" y2="135" stroke="black" marker-end="url(#arrow5)"/> <line x1="540" y1="135" x2="580" y2="55" stroke="black" marker-end="url(#arrow5)"/> <line x1="540" y1="135" x2="580" y2="135" stroke="black" marker-end="url(#arrow5)"/> <line x1="540" y1="135" x2="580" y2="215" stroke="black" marker-end="url(#arrow5)"/> </svg>`


Diagram 6. Pogo Stick Landing Concept (Visible vs Invisible Jumps)
Alt text: Two parallel sequences of four jumps; assistant mode shows user interactions at each landing, turbulence mode shows autonomous transitions.
`<svg xmlns="http://www.w3.org/2000/svg" width="920" height="300">
<defs>
<marker id="arrow6" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
<polygon points="0,0 10,5 0,10" fill="black"/>
</marker>
</defs>
<text x="170" y="30" font-size="14" text-anchor="middle">Assistant Mode (Visible Jumps)</text>
<text x="720" y="30" font-size="14" text-anchor="middle">Turbulence Mode (Invisible Jumps)</text>
<!-- Assistant jumps --> <rect x="40" y="50" width="120" height="50" fill="none" stroke="black"/> <text x="100" y="70" font-size="10" text-anchor="middle">Thermo Pixels</text> <text x="100" y="85" font-size="10" text-anchor="middle">+ Chat</text> <rect x="190" y="50" width="120" height="50" fill="none" stroke="black"/> <text x="250" y="70" font-size="10" text-anchor="middle">Bayesian</text> <text x="250" y="85" font-size="10" text-anchor="middle">+ Chat</text> <rect x="340" y="50" width="120" height="50" fill="none" stroke="black"/> <text x="400" y="70" font-size="10" text-anchor="middle">Reconstruction</text> <text x="400" y="85" font-size="10" text-anchor="middle">+ Chat</text> <rect x="490" y="50" width="120" height="50" fill="none" stroke="black"/> <text x="550" y="70" font-size="10" text-anchor="middle">Variance</text> <text x="550" y="85" font-size="10" text-anchor="middle">Confirmation</text> <line x1="160" y1="75" x2="190" y2="75" stroke="black" marker-end="url(#arrow6)"/> <line x1="310" y1="75" x2="340" y2="75" stroke="black" marker-end="url(#arrow6)"/> <line x1="460" y1="75" x2="490" y2="75" stroke="black" marker-end="url(#arrow6)"/> <rect x="640" y="50" width="120" height="50" fill="none" stroke="black"/> <text x="700" y="70" font-size="10" text-anchor="middle">BMD Extract</text> <rect x="790" y="50" width="120" height="50" fill="none" stroke="black"/> <text x="850" y="70" font-size="10" text-anchor="middle">Cross-Product</text> <rect x="640" y="130" width="120" height="50" fill="none" stroke="black"/> <text x="700" y="150" font-size="10" text-anchor="middle">S-Entropy</text> <rect x="790" y="130" width="120" height="50" fill="none" stroke="black"/> <text x="850" y="150" font-size="10" text-anchor="middle">Equilibrium +</text> <text x="850" y="165" font-size="10" text-anchor="middle">Validation</text> <line x1="760" y1="75" x2="790" y2="75" stroke="black" marker-end="url(#arrow6)"/> <line x1="700" y1="100" x2="700" y2="130" stroke="black" marker-end="url(#arrow6)"/> <line x1="850" y1="100" x2="850" y2="130" stroke="black" marker-end="url(#arrow6)"/> <rect x="40" y="130" width="570" height="110" fill="none" stroke="black" stroke-dasharray="5 5"/> <text x="325" y="155" font-size="11" text-anchor="middle">User Interactions Maintain Interpretability</text> <text x="325" y="175" font-size="10" text-anchor="middle">Variance Bounds Adjusted with Feedback</text> <rect x="640" y="200" width="270" height="40" fill="none" stroke="black" stroke-dasharray="5 5"/> <text x="775" y="225" font-size="10" text-anchor="middle">Autonomous Minimization (No Visible Jumps)</text> </svg>`


Diagram 7. Integration with Helicopter Framework
Alt text: Helicopter engine feeds image into Moon-Landing controller; assistant or turbulence outputs optionally validated by Helicopter validation.
`<svg xmlns="http://www.w3.org/2000/svg" width="880" height="240">
<defs>
<marker id="arrow7" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
<polygon points="0,0 10,5 0,10" fill="black"/>
</marker>
</defs>
<rect x="40" y="90" width="140" height="60" fill="none" stroke="black"/>
<text x="110" y="115" font-size="11" text-anchor="middle">Image /</text>
<text x="110" y="130" font-size="11" text-anchor="middle">Multi-Modal Input</text>

<rect x="210" y="80" width="170" height="80" fill="none" stroke="black"/> <text x="295" y="105" font-size="11" text-anchor="middle">Prepare Multi-Modal</text> <text x="295" y="120" font-size="11" text-anchor="middle">Input Adapter</text> <rect x="410" y="40" width="150" height="60" fill="none" stroke="black"/> <text x="485" y="65" font-size="11" text-anchor="middle">Assistant Path</text> <rect x="410" y="130" width="150" height="60" fill="none" stroke="black"/> <text x="485" y="160" font-size="11" text-anchor="middle">Turbulence Path</text> <rect x="600" y="85" width="140" height="70" fill="none" stroke="black"/> <text x="670" y="110" font-size="11" text-anchor="middle">Moon-Landing</text> <text x="670" y="125" font-size="11" text-anchor="middle">Unified Output</text> <rect x="770" y="85" width="90" height="70" fill="none" stroke="black"/> <text x="815" y="110" font-size="11" text-anchor="middle">Helicopter</text> <text x="815" y="125" font-size="11" text-anchor="middle">Validation</text> <line x1="180" y1="120" x2="210" y2="120" stroke="black" marker-end="url(#arrow7)"/> <line x1="380" y1="80" x2="410" y2="70" stroke="black" marker-end="url(#arrow7)"/> <line x1="380" y1="160" x2="410" y2="160" stroke="black" marker-end="url(#arrow7)"/> <line x1="560" y1="70" x2="600" y2="120" stroke="black" marker-end="url(#arrow7)"/> <line x1="560" y1="160" x2="600" y2="120" stroke="black" marker-end="url(#arrow7)"/> <line x1="740" y1="120" x2="770" y2="120" stroke="black" marker-end="url(#arrow7)"/> <rect x="410" y="10" width="150" height="20" fill="none" stroke="black" stroke-dasharray="4 4"/> <text x="485" y="25" font-size="10" text-anchor="middle">Mode Selection Internally</text> </svg>`

Diagram 8. Performance Metrics Snapshot
Alt text: Metrics boxes feeding success criteria box summarizing thresholds.
`<svg xmlns="http://www.w3.org/2000/svg" width="900" height="260">
<defs>
<marker id="arrow8" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
<polygon points="0,0 10,5 0,10" fill="black"/>
</marker>
</defs>
<rect x="40" y="40" width="160" height="60" fill="none" stroke="black"/>
<text x="120" y="65" font-size="11" text-anchor="middle">Variance</text>
<text x="120" y="80" font-size="11" text-anchor="middle">Reduction</text>

<rect x="240" y="40" width="160" height="60" fill="none" stroke="black"/> <text x="320" y="60" font-size="11" text-anchor="middle">Path Efficiency</text> <text x="320" y="75" font-size="10" text-anchor="middle">Steps to Equilibrium</text> <rect x="440" y="40" width="160" height="60" fill="none" stroke="black"/> <text x="520" y="60" font-size="11" text-anchor="middle">Mode Switching</text> <text x="520" y="75" font-size="10" text-anchor="middle">Accuracy</text> <rect x="640" y="40" width="160" height="60" fill="none" stroke="black"/> <text x="720" y="60" font-size="11" text-anchor="middle">Consciousness</text> <text x="720" y="75" font-size="10" text-anchor="middle">Validation Rate</text> <rect x="240" y="130" width="160" height="60" fill="none" stroke="black"/> <text x="320" y="150" font-size="11" text-anchor="middle">Human</text> <text x="320" y="165" font-size="11" text-anchor="middle">Comprehension</text> <rect x="440" y="130" width="160" height="60" fill="none" stroke="black"/> <text x="520" y="150" font-size="11" text-anchor="middle">Cross-Modal</text> <text x="520" y="165" font-size="11" text-anchor="middle">Convergence</text> <rect x="640" y="130" width="160" height="60" fill="none" stroke="black"/> <text x="720" y="150" font-size="11" text-anchor="middle">Processing</text> <text x="720" y="165" font-size="11" text-anchor="middle">Latency</text> <rect x="360" y="210" width="300" height="40" fill="none" stroke="black"/> <text x="510" y="230" font-size="11" text-anchor="middle">Success Criteria Thresholds Aggregation</text> <line x1="120" y1="100" x2="320" y2="210" stroke="black" marker-end="url(#arrow8)"/> <line x1="320" y1="100" x2="420" y2="210" stroke="black" marker-end="url(#arrow8)"/> <line x1="520" y1="100" x2="450" y2="210" stroke="black" marker-end="url(#arrow8)"/> <line x1="720" y1="100" x2="530" y2="210" stroke="black" marker-end="url(#arrow8)"/> <line x1="320" y1="190" x2="480" y2="210" stroke="black" marker-end="url(#arrow8)"/> <line x1="520" y1="190" x2="520" y2="210" stroke="black" marker-end="url(#arrow8)"/> <line x1="720" y1="190" x2="600" y2="210" stroke="black" marker-end="url(#arrow8)"/> </svg>`