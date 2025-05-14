document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('search-form');
    const searchButton = document.getElementById('search-button');
    const buttonText = searchButton.querySelector('.button-text');
    const spinner = searchButton.querySelector('.button-spinner'); // Corrected selector

    const photoInput = document.getElementById('photo');
    const photoPreview = document.getElementById('photo-preview');

    const inputSection = document.getElementById('input-section'); // Get input section
    const processingSection = document.getElementById('processing-section');
    const progressStepsList = document.getElementById('progress-steps-list'); // Corrected ID
    const resultsSection = document.getElementById('results-section');
    const resultsContentArea = document.getElementById('results-content-area');
    const noResultsMessage = document.getElementById('no-results-message');


    // Photo preview
    photoInput.addEventListener('change', function() {
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
            photoPreview.src = "#"; // Clear src
        }
    });

    searchForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        clearPreviousResults();
        showProcessingState(true);
        inputSection.classList.remove('active-section'); // Hide input section
        processingSection.classList.add('active-section'); // Show processing section

        const formData = new FormData(searchForm);
        const nameValue = document.getElementById('name').value;
        if (nameValue) {
            formData.set('name', nameValue);
        } else {
            formData.delete('name'); // Ensure 'name' is not sent if empty
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


        const stages = [
            "Uploading photo and name...",
            "Generating face embedding for input photo...",
            ...(nameValue ? ["Generating name embedding for input name..."] : []),
            "Searching database for similar faces (FAISS)...",
            ...(nameValue ? ["Searching database for similar names (FAISS)..."] : []),
            "Fusing search results (Reciprocal Rank Fusion)...",
            "Identifying candidate families from fused results...",
            "Classifying family membership for candidates...",
            "Predicting relationships to members in top families...",
            "Finalizing results..."
        ];

        stages.forEach(stage => addProgressStep(stage, 'pending', 'pending'));

        function updateStageStatus(index, statusText, statusClass) {
            const stepItem = progressStepsList.children[index];
            if (stepItem) {
                const statusSpan = stepItem.querySelector('.status-icon');
                statusSpan.textContent = '';
                statusSpan.className = `status-icon status-${statusClass}`;
            }
        }
        updateStageStatus(0, 'Working...', 'working');


        try {
            const response = await fetch('/query', {
                method: 'POST',
                body: formData,
            });

            for (let i = 0; i < stages.length -1; i++) {
                 await new Promise(resolve => setTimeout(resolve, 100));
                 updateStageStatus(i, 'Completed', 'success');
                 if (stages[i+1]) updateStageStatus(i + 1, 'Working...', 'working');
            }


            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Server returned an error, but response was not valid JSON.' }));
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            const results = await response.json();
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

    function showProcessingState(isProcessing) {
        if (isProcessing) {
            searchButton.disabled = true;
            buttonText.style.display = 'none';
            spinner.style.display = 'inline-block';
        } else {
            searchButton.disabled = false;
            buttonText.style.display = 'inline-block';
            spinner.style.display = 'none';
        }
    }

    function addProgressStep(text, statusText, statusClass) { // statusText not used here
        const listItem = document.createElement('li');
        listItem.textContent = text + " ";

        const statusSpan = document.createElement('span');
        statusSpan.className = `status-icon status-${statusClass}`;
        listItem.appendChild(statusSpan);

        progressStepsList.appendChild(listItem);
        processingSection.scrollTop = processingSection.scrollHeight;
    }

    function clearPreviousResults() {
        progressStepsList.innerHTML = '';
        resultsContentArea.innerHTML = '';
        noResultsMessage.style.display = 'none';
        resultsSection.classList.remove('active-section');
        processingSection.classList.remove('active-section');
        inputSection.classList.add('active-section'); // Default back to input
    }

   function displayResults(data) {
       resultsSection.classList.add('active-section'); // Ensure results section is visible
       resultsContentArea.innerHTML = ''; // Clear previous results from this area
       noResultsMessage.style.display = 'none'; // Hide no-results message initially

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
           personDiv.innerHTML = `<h3><i class="fas fa-user-check"></i> Strong Direct Match Identified!</h3>
               <p><strong>Person ID:</strong> ${data.identified_person.PersonID}</p>
               <p><strong>Name:</strong> ${data.identified_person.Name}</p>
               <p><strong>Family ID (FID):</strong> ${data.identified_person.FID}</p>
               <p><strong>Face Similarity Score:</strong> ${parseFloat(data.identified_person.face_similarity).toFixed(4)}</p>`;
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

                // Modified family header line
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

                       memberItem.innerHTML = `
                           <div class="member-info-container">
                               ${memberImageHtml}
                               <div class="member-text-details">
                                   <strong>${member.name}</strong> (ID: ${member.person_id})
                                   <br><em>Potential Relationship: ${member.relationship_to_input || 'N/A'}</em>
                               </div>
                           </div>`;
                       membersList.appendChild(memberItem);
                   });
                   familyCard.appendChild(membersList);
               } else if (family.message) {
                   const noMemberMsg = document.createElement('p');
                   noMemberMsg.textContent = family.message;
                   familyCard.appendChild(noMemberMsg);
               } else {
                   const noMemberMsg = document.createElement('p');
                   noMemberMsg.textContent = "No member details available for this family or relationship classifier was not run.";
                   familyCard.appendChild(noMemberMsg);
               }
               familiesContainer.appendChild(familyCard);
            });
            resultsContentArea.appendChild(familiesContainer);
        }

       if (!foundContent && !data.error) {
            noResultsMessage.style.display = 'block';
       }
   }

    function displayError(message, area) {
        const displayArea = area || resultsContentArea;
        displayArea.innerHTML = `<div class="error-card"><i class="fas fa-exclamation-triangle"></i><p class="error-message">${message}</p></div>`;
    }
});