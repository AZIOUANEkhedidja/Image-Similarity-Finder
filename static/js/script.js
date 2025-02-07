// Get the file input, drop zone, and preview container
const fileInput = document.getElementById('query');
const dropZone = document.getElementById('drop-zone');
const preview = document.getElementById('preview');
const section = document.getElementById('section');
// Function to handle file selection
function handleFile(file) {
    // Check if the file is an image
    if (file && file.type.startsWith('image/')) {
        // Create a FormData object
        const formData = new FormData();
        formData.append('file', file);

        // Send the image to the backend
        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
        .then((response) => response.json())
        .then((data) => {
            if (data.filepath) {
                preview.innerHTML = `<img src="${URL.createObjectURL(file)}" alt="Selected Image">`;
                dropZone.style.height = "300px";
                // alert("Image uploaded successfully!");
            } else {
                alert(data.error || "Error uploading image.");
            }
            data.filepath.forEach(image => {
                const imgElement = document.createElement('img');
                imgElement.style.width = "230px";
                imgElement.style.height = "auto";
                imgElement.style.margin = "10px";
                imgElement.style.borderRadius = "8px";
                imgElement.style.boxShadow = "2px 2px 10px rgba(0, 0, 0, 0.2)";
                imgElement.style.transition = "transform 0.3s ease";
                imgElement.src = image; // هنا بدلي المصدر تاع الصورة
                // Append image element and label to ImgSec
                section.appendChild(imgElement);
            });
        })
        .catch((error) => {
            console.error('Error:', error);
            alert("Failed to upload image.");
        });
    } else {
        alert("Please select an image!");
    }
}
// Handle file input change
fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    handleFile(file);
});

// Handle drag-and-drop events
dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('dragover');

    // Get the file from the drop event
    const file = event.dataTransfer.files[0];
    handleFile(file);
});

// Allow click on drop zone to trigger file input
dropZone.addEventListener('click', () => {
    fileInput.click();
});
