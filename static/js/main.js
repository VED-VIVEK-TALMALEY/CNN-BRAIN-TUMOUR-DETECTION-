/**
 * Main JavaScript for Brain Tumor Detection Application
 */

class BrainTumorDetector {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.currentFile = null;
    }

    initializeElements() {
        // DOM Elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.selectedFile = document.getElementById('selectedFile');
        this.fileName = document.getElementById('fileName');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.loadingSection = document.getElementById('loadingSection');
        this.errorSection = document.getElementById('errorSection');
        this.errorText = document.getElementById('errorText');
        this.resultsSection = document.getElementById('resultsSection');
        this.progressFill = document.getElementById('progressFill');
        
        // Result elements
        this.detectionResult = document.getElementById('detectionResult');
        this.confidence = document.getElementById('confidence');
        this.probability = document.getElementById('probability');
        this.analysisTime = document.getElementById('analysisTime');
    }

    bindEvents() {
        // File upload handling
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        // File input change
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Analyze button
        this.analyzeBtn.addEventListener('click', this.analyzeImage.bind(this));
        
        // Keyboard navigation
        this.uploadArea.addEventListener('keydown', this.handleKeydown.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave() {
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        if (e.target.files.length > 0) {
            this.processFile(e.target.files[0]);
        }
    }

    handleKeydown(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            this.fileInput.click();
        }
    }

    processFile(file) {
        if (this.validateFile(file)) {
            this.currentFile = file;
            this.fileName.textContent = file.name;
            this.selectedFile.classList.add('show');
            this.analyzeBtn.disabled = false;
            this.hideError();
            
            // Animate button
            this.animateButton();
        } else {
            this.showError('Please select a valid image file (JPG, PNG, JPEG, BMP).');
        }
    }

    validateFile(file) {
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp'];
        const maxSize = 16 * 1024 * 1024; // 16MB
        
        if (!allowedTypes.includes(file.type)) {
            return false;
        }
        
        if (file.size > maxSize) {
            return false;
        }
        
        return true;
    }

    animateButton() {
        this.analyzeBtn.style.animation = 'none';
        this.analyzeBtn.offsetHeight; // Trigger reflow
        this.analyzeBtn.style.animation = 'pulse 0.6s ease-in-out';
    }

    async analyzeImage() {
        if (!this.currentFile) return;

        this.showLoading();
        this.hideError();
        this.hideResults();
        
        // Simulate progress
        this.simulateProgress();
        
        try {
            const formData = new FormData();
            formData.append('file', this.currentFile);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            // Wait for computing animation to complete before showing results
            await this.waitForAnimationCompletion();
            
            this.hideLoading();
            
            if (data.error) {
                this.showError(data.error);
            } else {
                this.showResults(data);
            }
        } catch (err) {
            this.hideLoading();
            this.showError('Network error. Please try again.');
            console.error('Analysis error:', err);
        }
    }

    simulateProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
            }
            this.progressFill.style.width = progress + '%';
        }, 200);
    }

    showLoading() {
        this.loadingSection.classList.add('show');
        this.analyzeBtn.disabled = true;
        this.startComputingAnimation();
    }

    hideLoading() {
        this.loadingSection.classList.remove('show');
        this.analyzeBtn.disabled = false;
        this.progressFill.style.width = '0%';
        this.stopComputingAnimation();
    }

    showError(message) {
        this.errorText.textContent = message;
        this.errorSection.classList.add('show');
        this.errorSection.scrollIntoView({ behavior: 'smooth' });
    }

    hideError() {
        this.errorSection.classList.remove('show');
    }

    showResults(data) {
        // Animate results appearance
        setTimeout(() => {
            this.detectionResult.textContent = data.result;
            this.detectionResult.className = 'result-value ' + 
                (data.result === 'Tumor Detected' ? 'tumor-detected' : 'no-tumor');
        }, 100);

        setTimeout(() => {
            this.confidence.textContent = data.confidence;
        }, 200);

        setTimeout(() => {
            this.probability.textContent = data.probability;
        }, 300);

        setTimeout(() => {
            this.analysisTime.textContent = '~2.3s';
        }, 400);

        this.resultsSection.classList.add('show');
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    hideResults() {
        this.resultsSection.classList.remove('show');
    }

    startComputingAnimation() {
        // Create computing process visualization
        this.createComputingProcess();
        
        // Start neural network animation
        this.startNeuralNetworkAnimation();
        
        // Start data flow animation
        this.startDataFlowAnimation();
    }

    stopComputingAnimation() {
        // Stop all animations
        if (this.neuralNetworkInterval) {
            clearInterval(this.neuralNetworkInterval);
        }
        if (this.dataFlowInterval) {
            clearInterval(this.dataFlowInterval);
        }
        
        // Remove computing process elements
        this.removeComputingProcess();
    }

    createComputingProcess() {
        // Create computing process container
        this.computingProcess = document.createElement('div');
        this.computingProcess.className = 'computing-process';
        this.computingProcess.innerHTML = `
            <div class="process-header">
                <i class="fas fa-brain"></i>
                <span>AI Neural Network Processing</span>
            </div>
            <div class="process-steps">
                <div class="process-step active" data-step="1">
                    <i class="fas fa-image"></i>
                    <span>Image Preprocessing</span>
                    <div class="step-progress"></div>
                </div>
                <div class="process-step" data-step="2">
                    <i class="fas fa-cogs"></i>
                    <span>Feature Extraction</span>
                    <div class="step-progress"></div>
                </div>
                <div class="process-step" data-step="3">
                    <i class="fas fa-network-wired"></i>
                    <span>Neural Network Analysis</span>
                    <div class="step-progress"></div>
                </div>
                <div class="process-step" data-step="4">
                    <i class="fas fa-chart-line"></i>
                    <span>Result Classification</span>
                    <div class="step-progress"></div>
                </div>
            </div>
            <div class="neural-network-visualization">
                <div class="network-layers">
                    <div class="layer input-layer">
                        <div class="neuron"></div>
                        <div class="neuron"></div>
                        <div class="neuron"></div>
                    </div>
                    <div class="layer hidden-layer-1">
                        <div class="neuron"></div>
                        <div class="neuron"></div>
                        <div class="neuron"></div>
                        <div class="neuron"></div>
                    </div>
                    <div class="layer hidden-layer-2">
                        <div class="neuron"></div>
                        <div class="neuron"></div>
                        <div class="neuron"></div>
                    </div>
                    <div class="layer output-layer">
                        <div class="neuron"></div>
                    </div>
                </div>
                <div class="data-flow">
                    <div class="data-particle"></div>
                    <div class="data-particle"></div>
                    <div class="data-particle"></div>
                </div>
            </div>
            <div class="floating-particles">
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
                <div class="particle"></div>
            </div>
        `;
        
        this.loadingSection.appendChild(this.computingProcess);
        
        // Start step progression
        this.startStepProgression();
        
        // Start floating particles animation
        this.startFloatingParticles();
    }

    removeComputingProcess() {
        if (this.computingProcess) {
            this.computingProcess.remove();
            this.computingProcess = null;
        }
    }

    startStepProgression() {
        let currentStep = 1;
        const steps = document.querySelectorAll('.process-step');
        
        this.stepInterval = setInterval(() => {
            // Remove active class from all steps
            steps.forEach(step => step.classList.remove('active'));
            
            // Add active class to current step
            if (currentStep <= steps.length) {
                steps[currentStep - 1].classList.add('active');
                currentStep++;
            } else {
                clearInterval(this.stepInterval);
                // Add completion indicator
                this.showCompletionIndicator();
            }
        }, 800);
    }

    showCompletionIndicator() {
        // Add a completion checkmark to the last step
        const lastStep = document.querySelector('.process-step:last-child');
        if (lastStep) {
            const completionIcon = document.createElement('i');
            completionIcon.className = 'fas fa-check-circle completion-icon';
            completionIcon.style.color = '#10b981';
            completionIcon.style.fontSize = '1.2rem';
            completionIcon.style.marginTop = '0.5rem';
            completionIcon.style.animation = 'fadeInScale 0.5s ease-out';
            lastStep.appendChild(completionIcon);
        }
    }

    startNeuralNetworkAnimation() {
        const neurons = document.querySelectorAll('.neuron');
        let currentNeuron = 0;
        
        this.neuralNetworkInterval = setInterval(() => {
            if (currentNeuron < neurons.length) {
                neurons[currentNeuron].classList.add('active');
                currentNeuron++;
            } else {
                clearInterval(this.neuralNetworkInterval);
            }
        }, 200);
    }

    startDataFlowAnimation() {
        const particles = document.querySelectorAll('.data-particle');
        let currentParticle = 0;
        
        this.dataFlowInterval = setInterval(() => {
            if (currentParticle < particles.length) {
                particles[currentParticle].classList.add('flowing');
                currentParticle++;
            } else {
                clearInterval(this.dataFlowInterval);
            }
        }, 300);
    }

    startFloatingParticles() {
        const particles = document.querySelectorAll('.floating-particles .particle');
        particles.forEach((particle, index) => {
            particle.style.animationDelay = `${index * 0.5}s`;
            particle.classList.add('floating');
        });
    }

    waitForAnimationCompletion() {
        return new Promise((resolve) => {
            // Calculate total animation duration based on all steps
            const totalSteps = 4; // Image Preprocessing, Feature Extraction, Neural Network Analysis, Result Classification
            const stepDuration = 800; // 800ms per step
            const neuralNetworkDuration = 11 * 200; // 11 neurons * 200ms each
            const dataFlowDuration = 3 * 300; // 3 particles * 300ms each
            const floatingParticlesDuration = 3000; // 3 seconds for floating particles
            
            // Use the longest duration to ensure all animations complete
            const totalDuration = Math.max(
                totalSteps * stepDuration,
                neuralNetworkDuration,
                dataFlowDuration,
                floatingParticlesDuration
            ) + 1000; // Add 1 second buffer for smooth transition
            
            setTimeout(() => {
                resolve();
            }, totalDuration);
        });
    }

    // Utility methods
    resetForm() {
        this.currentFile = null;
        this.fileInput.value = '';
        this.selectedFile.classList.remove('show');
        this.analyzeBtn.disabled = true;
        this.hideError();
        this.hideResults();
    }

    // Accessibility improvements
    setFocusable() {
        this.uploadArea.setAttribute('tabindex', '0');
        this.uploadArea.setAttribute('role', 'button');
        this.uploadArea.setAttribute('aria-label', 'Upload MRI scan image');
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new BrainTumorDetector();
    
    // Add entrance animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe all sections for animations
    document.querySelectorAll('.upload-section, .instructions-section').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'all 0.6s ease-out';
        observer.observe(section);
    });
    
    // Set accessibility attributes
    app.setFocusable();
    
    // Global error handler
    window.addEventListener('error', (e) => {
        console.error('Global error:', e.error);
    });
    
    // Service Worker registration (for PWA capabilities)
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('ServiceWorker registration successful');
            })
            .catch(err => {
                console.log('ServiceWorker registration failed: ', err);
            });
    }
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BrainTumorDetector;
}
