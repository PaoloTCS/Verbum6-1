class VoronoiMap {
    constructor(containerId) {
        this.container = d3.select(`#${containerId}`);
        this.width = this.container.node().getBoundingClientRect().width;
        this.height = 800;
        this.padding = 60;
        this.currentPath = [];
        this.currentData = null;
        
        this.svg = this.container.append('svg')
            .attr('width', this.width)
            .attr('height', this.height);
        
        this.initialize();
        this.documentViewer = new DocumentViewer();
    }

    async initialize() {
        try {
            // Fetch both hierarchy and semantic distances
            const [hierarchyResponse, distancesResponse] = await Promise.all([
                fetch('/api/hierarchy'),
                fetch('/api/semantic-distances/level-0')
            ]);
            
            this.currentData = await hierarchyResponse.json();
            const distances = await distancesResponse.json();
            
            // Merge distances into hierarchy data
            this.currentData.hierarchy.distances = distances.distances;
            this.renderLevel(this.currentData.hierarchy);
        } catch (error) {
            console.error('Failed to load data:', error);
        }
    }

    generatePoints(data) {
        const children = data.children || [];
        if (children.length === 0) return [];

        if (data.distances) {
            return this.positionWithForceLayout(children, data.distances);
        }

        return this.circularLayout(children);
    }

    circularLayout(nodes) {
        const radius = Math.min(this.width, this.height) / 2.5;
        return nodes.map((d, i) => ({
            ...d,
            x: this.width/2 + radius * Math.cos((i / nodes.length) * 2 * Math.PI),
            y: this.height/2 + radius * Math.sin((i / nodes.length) * 2 * Math.PI)
        }));
    }

    positionWithForceLayout(nodes, distances) {
        const simulation = d3.forceSimulation(nodes)
            .force('charge', d3.forceManyBody().strength(-2000))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(100))
            .force('semantic', this.createSemanticForce(distances))
            .stop();

        for (let i = 0; i < 300; ++i) simulation.tick();
        return nodes;
    }

    createSemanticForce(distances) {
        return (alpha) => {
            Object.entries(distances).forEach(([pair, distance]) => {
                const [id1, id2] = pair.split('|');
                const node1 = this.currentData.hierarchy.children.find(n => n.name === id1);
                const node2 = this.currentData.hierarchy.children.find(n => n.name === id2);
                
                if (node1 && node2) {
                    const dx = node2.x - node1.x;
                    const dy = node2.y - node1.y;
                    const l = Math.sqrt(dx * dx + dy * dy);
                    const targetDistance = distance * 200;
                    
                    if (l !== 0) {
                        const force = (l - targetDistance) * alpha;
                        node1.x += dx * force / l;
                        node1.y += dy * force / l;
                        node2.x -= dx * force / l;
                        node2.y -= dy * force / l;
                    }
                }
            });
        };
    }

    renderLevel(data) {
        console.group('Rendering Level');
        console.log('Current path:', this.currentPath);
        console.log('Level data:', data);
        
        const points = this.generatePoints(data);
        console.log('Generated points:', points);
        
        if (points.length === 0) {
            console.warn('No points to render');
            console.groupEnd();
            return;
        }

        const delaunay = d3.Delaunay.from(points, d => d.x, d => d.y);
        const voronoi = delaunay.voronoi([
            this.padding, 
            this.padding, 
            this.width - this.padding, 
            this.height - this.padding
        ]);

        // Clear previous content
        this.svg.selectAll('*').remove();

        // Draw cells
        const cells = this.svg.append('g')
            .selectAll('g')
            .data(points)
            .join('g');

        cells.append('path')
            .attr('d', (_, i) => `M${voronoi.cellPolygon(i).join('L')}Z`)
            .attr('fill', (_, i) => d3.interpolateRainbow(i / points.length))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .on('click', (event, d) => this.handleCellClick(d));

        // Add labels
        cells.append('text')
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('fill', 'white')
            .style('pointer-events', 'none')
            .text(d => d.name);

        this.updateBreadcrumb();
        console.groupEnd();
    }

    handleCellClick(d) {
        console.group('Cell Click Handler');
        console.log('Clicked cell data:', d);
        
        if (d.type === 'document') {
            console.log('Document detected:', d.path);
            if (d.path.toLowerCase().endsWith('.pdf')) {
                console.log('PDF document detected, attempting to load');
                if (this.documentViewer) {
                    this.documentViewer.loadDocument(d.path);
                } else {
                    console.error('Document viewer not initialized');
                }
            } else {
                console.log('Non-PDF document detected:', d.path);
            }
        } else if (d.type === 'folder') {
            console.log('Folder detected:', d.name);
            this.currentPath.push(d.name);
            this.renderLevel(d);
        } else {
            console.log('Unknown cell type:', d);
        }
        
        console.groupEnd();
    }

    updateBreadcrumb() {
        const breadcrumb = d3.select('#breadcrumb');
        breadcrumb.html('');
        
        breadcrumb.append('span')
            .text('Root')
            .style('cursor', 'pointer')
            .on('click', () => this.navigateToRoot());

        this.currentPath.forEach((path, i) => {
            breadcrumb.append('span').text(' > ');
            breadcrumb.append('span')
                .text(path)
                .style('cursor', 'pointer')
                .on('click', () => this.navigateToLevel(i));
        });
    }

    navigateToRoot() {
        this.currentPath = [];
        this.renderLevel(this.currentData.hierarchy);
    }

    navigateToLevel(level) {
        this.currentPath = this.currentPath.slice(0, level + 1);
        let currentNode = this.currentData.hierarchy;
        
        for (const pathSegment of this.currentPath) {
            currentNode = currentNode.children.find(c => c.name === pathSegment);
        }
        
        this.renderLevel(currentNode);
    }
}

class DocumentViewer {
    constructor() {
        this.panel = document.getElementById('document-panel');
        if (!this.panel) {
            console.error('Document panel element not found');
            return;
        }

        // Initialize panel content
        this.panel.innerHTML = `
            <div class="document-controls">
                <button class="control-button" id="minimize-doc">Minimize</button>
                <button class="control-button" id="close-doc">Close</button>
            </div>
            <div id="document-viewer"></div>
            <div id="qa-interface" class="qa-container">
                <div class="qa-input">
                    <input type="text" id="question-input" placeholder="Ask a question about this document...">
                    <button id="ask-button">Ask</button>
                </div>
                <div id="answer-display"></div>
            </div>
        `;

        this.initializeControls();
        this.initializeQA();
    }

    initializeControls() {
        const minimizeBtn = document.getElementById('minimize-doc');
        const closeBtn = document.getElementById('close-doc');
        const viewer = document.getElementById('document-viewer');

        if (minimizeBtn) {
            minimizeBtn.addEventListener('click', () => {
                if (viewer) {
                    viewer.style.height = viewer.style.height === '30vh' ? '70vh' : '30vh';
                }
            });
        }

        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                if (this.panel) {
                    this.panel.classList.add('hidden');
                }
            });
        }
    }

    initializeQA() {
        const askButton = document.getElementById('ask-button');
        const questionInput = document.getElementById('question-input');
        
        if (askButton && questionInput) {
            console.log('Initializing Q&A handlers');
            
            askButton.addEventListener('click', () => {
                const currentDoc = this.currentDocument;
                if (currentDoc) {
                    handleQuestionSubmit(currentDoc);
                }
            });

            questionInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    const currentDoc = this.currentDocument;
                    if (currentDoc) {
                        handleQuestionSubmit(currentDoc);
                    }
                }
            });
        }
    }

    async loadDocument(path) {
        if (!this.panel) return;
        
        try {
            console.log('Document Loading:', path);
            this.currentDocument = path;  // Store current document path
            
            this.panel.classList.remove('hidden');
            const viewer = document.getElementById('document-viewer');
            
            if (viewer) {
                const pdfUrl = `/api/document/${encodeURIComponent(path)}?t=${Date.now()}`;
                viewer.innerHTML = `<embed src="${pdfUrl}" type="application/pdf" width="100%" height="100%">`;
            }
        } catch (error) {
            console.error('PDF loading error:', error);
        }
    }
}

// Add this to your document load handler
function handleDocumentLoad(docPath) {
    // Show QA interface
    document.getElementById('qa-interface').style.display = 'block';
    document.getElementById('current-doc').textContent = `Current document: ${docPath}`;
    
    // Setup QA handlers
    initializeQAInterface(docPath);
}

async function handleQuestionSubmit(docPath) {
    console.log('handleQuestionSubmit called with:', docPath);
    const questionInput = document.getElementById('question-input');
    const answerDisplay = document.getElementById('answer-display');
    const question = questionInput.value.trim();
    
    if (!question) {
        console.log('No question entered');
        return;
    }
    
    console.log('Processing question:', question);
    answerDisplay.innerHTML = '<em>Processing question...</em>';
    
    try {
        console.log('Sending request to /api/ask');
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                document: docPath
            })
        });
        
        console.log('Response received:', response.status);
        const result = await response.json();
        console.log('Result:', result);
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        answerDisplay.innerHTML = `
            <div class="answer-content">
                <p><strong>${result.answer}</strong></p>
                <p class="confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</p>
            </div>
        `;
    } catch (error) {
        console.error('Error:', error);
        answerDisplay.innerHTML = `
            <div class="error-message">
                ${error.message}<br>
                <small>Please try again or contact support.</small>
            </div>
        `;
    }
}

// Update the document viewer initialization
function initializeQAInterface(docPath) {
    const askButton = document.getElementById('ask-button');
    if (askButton) {
        askButton.onclick = () => handleQuestionSubmit(docPath);
        
        // Also handle Enter key in input
        document.getElementById('question-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleQuestionSubmit(docPath);
            }
        });
    }
}

// Initialize the visualization and document viewer
document.addEventListener('DOMContentLoaded', () => {
    const voronoiMap = new VoronoiMap('visualization');
    window.documentViewer = new DocumentViewer();
});