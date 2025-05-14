document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('search-form');
    const searchButton = document.getElementById('search-button');
    const buttonText = searchButton.querySelector('.button-text');
    const spinner = searchButton.querySelector('.button-spinner');

    const photoInput = document.getElementById('photo');
    const photoPreview = document.getElementById('photo-preview');

    const inputSection = document.getElementById('input-section');
    const processingSection = document.getElementById('processing-section');
    const progressStepsList = document.getElementById('progress-steps-list');
    const resultsSection = document.getElementById('results-section');
    const resultsContentArea = document.getElementById('results-content-area');
    const noResultsMessage = document.getElementById('no-results-message');

    // Explanation Modal Elements
    const explanationModal = document.getElementById('explanation-modal');
    const explanationLoading = document.getElementById('explanation-loading');
    const explanationError = document.getElementById('explanation-error');
    const explanationImagesContainer = document.getElementById('explanation-images-container');
    const queryHeatmapImg = document.getElementById('query-heatmap-img');
    const memberHeatmapImg = document.getElementById('member-heatmap-img');

    let currentQueryPhotoFilename = null; // To store the uploaded photo's filename for explanations


    photoInput.addEventListener('change', function() {
        // ... (existing photo preview logic) ...
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                photoPreview.src = e.target.result;
                photoPreview.style.display = 'block';
            }
            reader.readAsDataURL(file);
        } else {
            photoPreview.style.display = 'none';
            photoPreview.src = "#";
        }
    });

    searchForm.addEventListener('submit', async (event) => {
        // ... (existing submit logic, stage updates) ...
        event.preventDefault();
        clearPreviousResults();
        showProcessingState(true);
        inputSection.classList.remove('active-section');
        processingSection.classList.add('active-section');

        const formData = new FormData(searchForm);
        const nameValue = document.getElementById('name').value;
        if (nameValue) {
            formData.set('name', nameValue);
        } else {
            formData.delete('name');
        }

        const photoFile = photoInput.files[0];
        if (!photoFile) {
            addProgressStep('Photo selection', 'Error: No photo selected.', 'error');
            showProcessingState(false);
            processingSection.classList.remove('active-section');
            inputSection.classList.add('active-section');
            return;
        }
        formData.set('photo', photoFile);

        const stages = [ /* ... existing stages ... */ ];
        stages.forEach(stage => addProgressStep(stage, 'pending', 'pending'));
        function updateStageStatus(index, statusText, statusClass) { /* ... */ }
        updateStageStatus(0, 'Working...', 'working');


        try {
            const response = await fetch('/query', {
                method: 'POST',
                body: formData,
            });

            for (let i = 0; i < stages.length -1; i++) {
                 await new Promise(resolve => setTimeout(resolve, 100)); // Simulate work
                 updateStageStatus(i, 'Completed', 'success');
                 if (stages[i+1]) updateStageStatus(i + 1, 'Working...', 'working');
            }

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Server returned an error, but response was not valid JSON.' }));
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            const results = await response.json();
            if (results.uploaded_query_photo_filename) { // Store for explanation requests
                currentQueryPhotoFilename = results.uploaded_query_photo_filename;
            } else {
                currentQueryPhotoFilename = null;
            }

            updateStageStatus(stages.length - 1, 'Completed', 'success');
            displayResults(results);

        } catch (error) {
            console.error('Search error:', error);
            if(stages.length > 0) updateStageStatus(stages.length - 1, `Error: ${error.message}`, 'error');
            displayError(error.message, resultsContentArea);
        } finally {
            showProcessingState(false);
            processingSection.classList.remove('active-section');
            resultsSection.classList.add('active-section');
        }
    });

    function showProcessingState(isProcessing) { /* ... existing ... */ }
    function addProgressStep(text, statusText, statusClass) { /* ... existing ... */ }
    function clearPreviousResults() { /* ... existing ... */ currentQueryPhotoFilename = null; }

    function displayResults(data) {
       // ... (existing result display logic) ...
       // Key modification: Add "Explain" button and data attributes to it
       resultsSection.classList.add('active-section');
       resultsContentArea.innerHTML = '';
       noResultsMessage.style.display = 'none';

       if (data.error) {
           displayError(data.error, resultsContentArea);
           return;
       }

       let foundContent = false;

       if (data.message && !(data.identified_person || (data.candidate_families && data.candidate_families.length > 0))) {
           const msgDiv = document.createElement('p');
           msgDiv.textContent = data.message;
           resultsContentArea.appendChild(msgDiv);
           foundContent = true;
       }

       if (data.identified_person) {
           foundContent = true;
           const personDiv = document.createElement('div');
           personDiv.className = 'result-block identified-person-card';
           let explainButtonHtml = '';
           if (currentQueryPhotoFilename && data.identified_person.photo_path) {
                explainButtonHtml = `<button class="explain-button"
                                        data-member-photopath-abs="${data.identified_person.photo_path}">
                                        <i class="fas fa-wand-magic-sparkles"></i> Explain Similarity
                                     </button>`;
           }
           personDiv.innerHTML = `<h3><i class="fas fa-user-check"></i> Strong Direct Match Identified!</h3>
               <p><strong>Person ID:</strong> ${data.identified_person.PersonID}</p>
               <p><strong>Name:</strong> ${data.identified_person.Name}</p>
               <p><strong>Family ID (FID):</strong> ${data.identified_person.FID}</p>
               <p><strong>Face Similarity Score:</strong> ${parseFloat(data.identified_person.face_similarity).toFixed(4)}</p>
               ${explainButtonHtml}`;
           resultsContentArea.appendChild(personDiv);
       }

       if (data.candidate_families && data.candidate_families.length > 0) {
            foundContent = true;
            const familiesContainer = document.createElement('div');
            familiesContainer.className = 'families-container';
            familiesContainer.innerHTML = `<h3><i class="fas fa-users"></i> Potential Candidate Families</h3>`;

            data.candidate_families.forEach(family => {
                const familyCard = document.createElement('div');
                familyCard.className = 'family-card result-block';
                familyCard.innerHTML = `<h4>
                                            ${family.family_display_name}
                                            <span class="family-id-muted">(${family.family_id})</span>
                                        </h4>
                    <p><strong>Probability of you belonging to this family:</strong> ${family.probability_belongs}</p>`;

                if (family.members && family.members.length > 0) {
                   const membersList = document.createElement('ul');
                   membersList.className = 'family-members-list';
                   family.members.forEach(member => {
                       const memberItem = document.createElement('li');
                       memberItem.className = 'family-member-item';

                       let memberImageHtml = '';
                       if (member.photo_url) {
                           memberImageHtml = `<img src="${member.photo_url}" alt="Photo of ${member.name}" class="member-photo" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                                            <div class="member-photo-placeholder" style="display:none;"><i class="fas fa-user-circle"></i></div>`;
                       } else {
                           memberImageHtml = `<div class="member-photo-placeholder"><i class="fas fa-user-circle"></i></div>`;
                       }

                       let memberExplainButtonHtml = '';
                       if (currentQueryPhotoFilename && member.photo_path_abs) {
                            memberExplainButtonHtml = `<button class="explain-button"
                                                            data-member-photopath-abs="${member.photo_path_abs}">
                                                            <i class="fas fa-wand-magic-sparkles"></i> Explain
                                                         </button>`;
                       }

                       memberItem.innerHTML = `
                           <div class="member-info-container">
                               ${memberImageHtml}
                               <div class="member-text-details">
                                   <strong>${member.name}</strong> (ID: ${member.person_id})
                                   <br><em>Potential Relationship: ${member.relationship_to_input || 'N/A'}</em>
                               </div>
                               ${memberExplainButtonHtml}
                           </div>`;
                       membersList.appendChild(memberItem);
                   });
                   familyCard.appendChild(membersList);
               } else if (family.message) { /* ... existing ... */ }
               else { /* ... existing ... */ }
               familiesContainer.appendChild(familyCard);
            });
            resultsContentArea.appendChild(familiesContainer);
        }

       if (!foundContent && !data.error) {
            noResultsMessage.style.display = 'block';
       }
       addExplainButtonListeners(); // Add listeners after new buttons are in DOM
   }

    function displayError(message, area) { /* ... existing ... */ }

    // --- Explanation Modal Logic ---
    function openModal() {
        explanationModal.style.display = "block";
        explanationLoading.style.display = "block";
        explanationImagesContainer.style.display = "none";
        explanationError.style.display = "none";
        queryHeatmapImg.src = "#"; // Clear previous
        memberHeatmapImg.src = "#"; // Clear previous
    }

    window.closeModal = function() { // Make it globally accessible for onclick
        explanationModal.style.display = "none";
    }

    window.onclick = function(event) { // Close modal if clicked outside
        if (event.target == explanationModal) {
            closeModal();
        }
    }

    function addExplainButtonListeners() {
        const explainButtons = document.querySelectorAll('.explain-button');
        explainButtons.forEach(button => {
            button.addEventListener('click', async function() {
                if (!currentQueryPhotoFilename) {
                    alert("Query photo information is missing. Please try searching again.");
                    return;
                }
                const memberPhotoPathAbs = this.dataset.memberPhotopathAbs;
                if (!memberPhotoPathAbs) {
                    alert("Member photo path is missing.");
                    return;
                }

                openModal(); // Show modal with loading state

                try {
                    const response = await fetch('/explain_similarity', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            query_photo_filename: currentQueryPhotoFilename,
                            member_photo_path_abs: memberPhotoPathAbs
                        }),
                    });

                    explanationLoading.style.display = "none";
                    if (!response.ok) {
                        const errData = await response.json().catch(() => ({error: "Server error during explanation."}));
                        throw new Error(errData.error || `Error ${response.status}`);
                    }

                    const data = await response.json();
                    if (data.error) {
                        throw new Error(data.error);
                    }

                    queryHeatmapImg.src = data.query_heatmap_url;
                    memberHeatmapImg.src = data.member_heatmap_url;
                    explanationImagesContainer.style.display = "flex"; // Show images

                } catch (error) {
                    console.error("Explanation error:", error);
                    explanationLoading.style.display = "none";
                    explanationError.querySelector('p').textContent = `Failed to generate explanation: ${error.message}`;
                    explanationError.style.display = "block";
                }
            });
        });
    }
});