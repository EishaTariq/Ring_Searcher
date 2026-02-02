"""
Flask Web Application for Ring Visual Search
============================================
Beautiful web interface for uploading and searching rings
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import json
from ring_visual_search import RingVisualSearch

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
CATALOG_FOLDER = 'catalog'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary folders
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(CATALOG_FOLDER).mkdir(exist_ok=True)

# Initialize search engine
search_engine = RingVisualSearch(catalog_path=CATALOG_FOLDER)

# Try to load existing catalog or build new one
print("\n🚀 Initializing Ring Visual Search System...")
if not search_engine._load_features_db():
    print("📚 Building catalog index...")
    search_engine.build_catalog_index()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/catalog-info')
def catalog_info():
    """Get catalog information"""
    catalog_files = list(Path(CATALOG_FOLDER).glob('*'))
    catalog_images = [f for f in catalog_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']]
    
    return jsonify({
        'total_rings': len(catalog_images),
        'indexed_rings': len(search_engine.catalog_features),
        'catalog_path': CATALOG_FOLDER
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and search"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Perform visual search - TOP 3 ONLY with STRONG deduplication
            top_k = 3  # FIXED: Always show only top 3 unique designs
            deduplicate = True  # ALWAYS enabled
            threshold = 85.0  # LOWERED threshold for stronger grouping
            
            results = search_engine.search(
                filepath, 
                top_k=top_k,
                deduplicate=deduplicate,
                similarity_threshold=threshold
            )
            
            # Format results for frontend
            formatted_results = []
            for match in results:
                result_data = {
                    'index': match['index'],
                    'similarity': match['match_percentage'],
                    'image_url': f"/catalog-image/{Path(match['image_path']).name}",
                    'filename': match['metadata']['filename']
                }
                
                # Add duplicate count if available
                if 'duplicate_count' in match:
                    result_data['duplicate_count'] = match['duplicate_count']
                
                # Add quality score if available
                if 'quality_score' in match:
                    result_data['quality_score'] = match['quality_score']
                
                formatted_results.append(result_data)
            
            return jsonify({
                'success': True,
                'query_image': f"/uploads/{filename}",
                'matches': formatted_results,
                'total_catalog': len(search_engine.catalog_features),
                'deduplicated': deduplicate,
                'unique_designs': len(results)
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/customize/<int:ring_index>')
def get_customization(ring_index):
    """Get customization options for selected ring"""
    try:
        options = search_engine.get_customization_options(ring_index)
        return jsonify(options)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rebuild-index', methods=['POST'])
def rebuild_index():
    """Rebuild catalog index"""
    try:
        search_engine.build_catalog_index()
        return jsonify({
            'success': True,
            'message': 'Catalog index rebuilt successfully',
            'total_rings': len(search_engine.catalog_features)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/catalog-image/<path:filename>')
def catalog_image(filename):
    """Serve catalog images"""
    return send_from_directory(CATALOG_FOLDER, filename)


@app.route('/catalog-preview')
def catalog_preview():
    """Get catalog preview images"""
    try:
        catalog_files = list(Path(CATALOG_FOLDER).glob('**/*'))
        catalog_images = [f for f in catalog_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']]
        
        # Get first 12 images as preview
        preview_images = []
        for img in catalog_images[:12]:
            preview_images.append({
                'filename': img.name,
                'url': f"/catalog-image/{img.relative_to(CATALOG_FOLDER)}",
                'path': str(img)
            })
        
        return jsonify({
            'total': len(catalog_images),
            'preview': preview_images
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌐 Starting Flask Web Server...")
    print("=" * 60)
    print("\n📍 Access the application at: http://localhost:5000")
    print("\n💡 Tips:")
    print("   - Add ring images to the 'catalog' folder")
    print("   - Click 'Rebuild Index' after adding new images")
    print("   - Upload ring images to find similar designs")
    print("\n" + "=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)