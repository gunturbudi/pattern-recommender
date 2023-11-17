from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_bootstrap import Bootstrap
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, UserMixin, current_user
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy import desc
from flask_migrate import Migrate
from tqdm import tqdm
import numpy as np
from werkzeug.wrappers import Response as ResponseBase

# from sentence_transformers import SentenceTransformer

# Initialize the sentence-transformers model
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///ltr_annotation.db'
app.config['SECRET_KEY'] = 'klaf9897fwehkwe' # Replace this with a real secret key

Bootstrap(app)
db = SQLAlchemy(app)


migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    annotations = relationship("Annotation", back_populates="user")

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    
query_tags = db.Table('query_tags',
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True),
    db.Column('query_id', db.Integer, db.ForeignKey('inquery.id'), primary_key=True)
)

candidate_tags = db.Table('candidate_tags',
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True),
    db.Column('candidate_id', db.Integer, db.ForeignKey('candidate.id'), primary_key=True)
)

class Inquery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(200), nullable=False)
    source = db.Column(db.String(200), nullable=True)
    tags = db.relationship('Tag', secondary=query_tags, lazy='subquery',
        backref=db.backref('inqueries', lazy=True))

    
class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(200), nullable=False)
    description = db.Column(db.String(1000), nullable=False)
    source = db.Column(db.String(200), nullable=True)
    tags = db.relationship('Tag', secondary=candidate_tags, lazy='subquery',
        backref=db.backref('candidates', lazy=True))

class CandidatesRecommended(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    query_id = db.Column(db.Integer, db.ForeignKey('inquery.id'), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidate.id'), nullable=False)
    relevance = db.Column(db.Float, nullable=False)
    query_data = relationship("Inquery", backref="recommended_candidates")
    candidate = relationship("Candidate")

class Annotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query_id = db.Column(db.Integer, db.ForeignKey('inquery.id'), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidate.id'), nullable=False)
    rank = db.Column(db.Integer, nullable=False)
    relevance = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="annotations")
    query_data = relationship("Inquery")
    candidate = relationship("Candidate")

class Similarity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    inquery_id = db.Column(db.Integer, db.ForeignKey('inquery.id'), nullable=False)
    other_inquery_id = db.Column(db.Integer, nullable=False)
    score = db.Column(db.Float, nullable=False)
    
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
@login_required
def home():
    total_queries_by_source = db.session.query(Inquery.source, func.count(Inquery.id)).group_by(Inquery.source).all()

    # Create a subquery that groups by query_id and only select query_id
    subquery = db.session.query(Annotation.query_id).filter(Annotation.user_id==current_user.id).group_by(Annotation.query_id).subquery()

    # Now group the subquery result by source
    total_queries_done_by_user = db.session.query(Inquery.source, func.count(subquery.c.query_id)).join(subquery, Inquery.id == subquery.c.query_id).group_by(Inquery.source).all()

    total_queries = {source: count for source, count in total_queries_by_source}
    queries_done = {source: count for source, count in total_queries_done_by_user}

    # This creates a list of dictionaries, where each dictionary contains the source, total queries, and queries done by the user
    queries_data = [{'source': source, 'total': total_queries[source], 'done': queries_done.get(source, 0)} for source in total_queries]

    return render_template('home.html', annotations_data=queries_data)

@app.route('/tag/<tag_name>')
def tag(tag_name):
    # Inquery the database to get all candidates with this tag
    tag = Tag.query.filter_by(name=tag_name).first_or_404()
    candidates = tag.candidates  # assuming a backref in your Tag model
    return render_template('tag.html', tag=tag, candidates=candidates)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.password == request.form['password']: # You should use hashed passwords in a real application
            login_user(user)
            return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        new_user = User(username=request.form['username'], password=request.form['password'])
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

def calculate_tag_overlap(query_tags, other_query_tags):
    return len(set(query_tags) & set(other_query_tags))

@app.route('/apply_annotation', methods=['POST'])
@login_required
def apply_annotation():
    # Fetch the current user's id
    user_id = current_user.id
    
    # Extract the query_id from the pre-annotated data and current_query_id from the form data
    previous_query_id = request.form.get("previous_query_id")
    current_query_id = request.form.get("current_query_id")

    sql = """
    INSERT INTO annotation (user_id, query_id, candidate_id, rank, relevance, timestamp)
    SELECT :user_id, :current_query_id, candidate_id, rank, relevance, datetime('now')
    FROM annotation
    WHERE query_id = :previous_query_id;
    """

    params = {
        "user_id": user_id,
        "current_query_id": current_query_id,
        "previous_query_id": previous_query_id
    }

    db.session.execute(sql, params)
    db.session.commit()

    
    return redirect(url_for('annotate', source=request.form.get("source")))


@app.route('/annotate/<source>', methods=['GET', 'POST'])
@login_required
def annotate(source):
    if request.method == 'POST':
        _save_annotations(request.form, current_user.id)
        
        return redirect(url_for('annotate', source=source))
        
    else:
        query = _get_unannotated_query_for_user(current_user.id, source)
        if not query:
            flash("All queries have been annotated. Thank you for your contribution!", "info")
            return redirect(url_for('home'))

        progress = _calculate_annotation_progress(source, current_user.id)
        similar_inquery_data = get_recommended_queries_by_semantic_similarity(query)
        recommended_candidates, other_candidates = _get_candidates(query, current_user.id)

        return render_template('annotate.html', query=query, similar_inquery_data=similar_inquery_data, 
                               recommended_candidates=recommended_candidates, other_candidates=other_candidates, 
                               progress=progress)

def _save_annotations(form_data, user_id):
    for candidate_id, relevance in form_data.items():
        if candidate_id == 'query_id':  # ignore the query text field
            continue
        annotation = Annotation(user_id=user_id, query_id=form_data["query_id"], candidate_id=candidate_id, 
                                relevance=float(relevance), rank=int(relevance))
        db.session.add(annotation)
    db.session.commit()

def _get_unannotated_query_for_user(user_id, source):
    annotated_queries_ids = db.session.query(Annotation.query_id).join(Inquery, Annotation.query_id == Inquery.id)\
        .filter(Annotation.user_id == user_id, Inquery.source == source).distinct()
    return Inquery.query.filter(Inquery.source == source, Inquery.id.notin_(annotated_queries_ids)).first()

def _get_candidates(query, user_id):
    recommended_candidates = CandidatesRecommended.query.filter_by(query_id=query.id).all()
    recommended_candidates_ids = [candidate.candidate_id for candidate in recommended_candidates]
    
    annotated_candidates_ids = [annotation.candidate_id for annotation in 
                                Annotation.query.filter_by(query_id=query.id, user_id=user_id).all()]
    
    other_candidates = Candidate.query.filter(Candidate.id.notin_(recommended_candidates_ids), 
                                              Candidate.id.notin_(annotated_candidates_ids)).all()
    return recommended_candidates, other_candidates

def _calculate_annotation_progress(source, user_id):
    total_queries = Inquery.query.filter_by(source=source).count()
    annotated_queries_ids = db.session.query(Annotation.query_id).join(Inquery, Annotation.query_id == Inquery.id)\
        .filter(Annotation.user_id == user_id, Inquery.source == source).distinct()
    annotated_queries_count = annotated_queries_ids.count()
    return (annotated_queries_count / total_queries) * 100

def get_recommended_queries_by_tags(query):
    query_tags = [tag.name for tag in query.tags]

def get_recommended_queries_by_semantic_similarity(query):
    query_tags = [tag.name for tag in query.tags]
    
    similar_inqueries_with_overlap = []
    
    # After fetching the query to annotate
    threshold = 0.4
    similarities = (
        Similarity.query.filter(Similarity.inquery_id == query.id, Similarity.score >= threshold)
        .order_by(desc(Similarity.score))
        .all()
    )
    for similarity in similarities:
        other_inquery = Inquery.query.get(similarity.other_inquery_id)
        other_query_tags = [tag.name for tag in other_inquery.tags]
        overlap_count = calculate_tag_overlap(query_tags, other_query_tags)
        similar_inqueries_with_overlap.append((similarity, overlap_count))

    # Order the inqueries based on the number of overlapping tags (in descending order)
    similar_inqueries_with_overlap.sort(key=lambda x: x[1], reverse=True)
    
    similar_inquery_data_dict = {}
    for similarity, _ in similar_inqueries_with_overlap:
        annotations = Annotation.query.filter_by(query_id=similarity.other_inquery_id).all()
        if annotations:
            for annotation in annotations:
                candidate = Candidate.query.get(annotation.candidate_id)
                if candidate:
                    other_inquery = Inquery.query.get(similarity.other_inquery_id)
                    tags = [tag.name for tag in other_inquery.tags]

                    if similarity.other_inquery_id not in similar_inquery_data_dict:
                        similar_inquery_data_dict[similarity.other_inquery_id] = {
                            "id": other_inquery.id,
                            "query": other_inquery.text,
                            "candidates": [(candidate.text, annotation.relevance)],
                            "timestamp": annotation.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            "tags": tags,
                            "score": "{:.2f}".format(similarity.score)
                        }
                    else:
                        similar_inquery_data_dict[similarity.other_inquery_id]['candidates'].append(
                            (candidate.text, annotation.relevance)
                        )

    # Convert dictionary to a list of dictionaries for rendering in the template
    similar_inquery_data = list(similar_inquery_data_dict.values())
    
    return similar_inquery_data
            
@app.route('/user_stats')
@login_required
def user_stats():
    user_annotations = Annotation.query.filter_by(user_id=current_user.id).all()
    annotations_data = {}

    for annotation in user_annotations:
        try:
            if annotation.query_data.id not in annotations_data:
                annotations_data[annotation.query_data.id] = {
                    'query': annotation.query_data.text,
                    'candidates': [(annotation.candidate.text, annotation.relevance)],
                    'timestamp': annotation.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                annotations_data[annotation.query_data.id]['candidates'].append(
                    (annotation.candidate.text, annotation.relevance))
        except AttributeError as e:
            app.logger.error(f'AttributeError for annotation id {annotation.id}: {str(e)}')

    # Convert dictionary to a list of dictionaries for easier handling in the template
    annotations_data = list(annotations_data.values())

    return render_template('user_stats.html', annotations_data=annotations_data)

@app.route('/precompute_similarity')
def precompute_similarity():
    # Retrieve all inqueries
    inqueries = Inquery.query.all()
    
    # Extract texts from inqueries
    texts = [inquery.text for inquery in inqueries]
    
    # Get embeddings for the texts
    embeddings = model.encode(texts)
    
    # Compute pairwise semantic similarity scores
    similarity_matrix = np.inner(embeddings, embeddings)
    
    # Store the precomputed similarities in the Similarity table
    for i in tqdm(range(len(inqueries)), desc="Storing Similarities"):
        inquery = inqueries[i]
        for j, other_inquery in enumerate(inqueries):
            if i != j:  # Skip self-similarity
                similarity = Similarity(inquery_id=inquery.id, other_inquery_id=other_inquery.id, score=similarity_matrix[i, j])
                db.session.add(similarity)
    db.session.commit()
    
def get_data_for_all_inqueries():
    # Fetch all inqueries
    inqueries = Inquery.query.all()
    
    all_data = []

    for inquery in inqueries:
        # Fetch all candidates
        candidates = Candidate.query.all()
        
        pattern_list = []
        
        for candidate in candidates:
            # Check if there's an annotation for the current inquery-candidate pair
            annotation = Annotation.query.filter_by(query_id=inquery.id, candidate_id=candidate.id).first()
            
            if annotation:
                relevance = annotation.relevance
            else:
                relevance = 0

            pattern_data = {
                "name": candidate.text,
                "rating": int(relevance)
            }
            pattern_list.append(pattern_data)

        # Format data for this inquery
        data = {
            "id": inquery.id,
            "req_type": [tag.name for tag in inquery.tags],  # Convert to list
            "req_name": f"{inquery.source} {inquery.id}",
            "req_text": inquery.text,
            "pattern": pattern_list
        }

        all_data.append(data)

    return all_data


@app.route('/download_all')
def download_json():
    data = get_data_for_all_inqueries()
    if not data:
        return "No inqueries found", 404

    # Convert the list of dictionaries to a JSON string
    json_data = jsonify(data).get_data(as_text=True)
    
    # Create a Flask Response with headers for download
    response = ResponseBase(json_data, content_type="application/json")
    response.headers["Content-Disposition"] = "attachment; filename=all_inqueries.json"
    return response

if __name__ == '__main__':
    app.run(debug=True)

