CREATE EXTENSION IF NOT EXISTS vector;


CREATE TABLE benchmark_queries (
	id VARCHAR NOT NULL,
	query TEXT NOT NULL,
	"case" VARCHAR NOT NULL,
	PRIMARY KEY (id)
)

;

CREATE TABLE evaluation_runs (
	id VARCHAR NOT NULL,
	created_at VARCHAR NOT NULL,
	results JSON NOT NULL,
	PRIMARY KEY (id)
)

;

CREATE TABLE investigations (
	id VARCHAR NOT NULL,
	title VARCHAR NOT NULL,
	query TEXT NOT NULL,
	created_at VARCHAR NOT NULL,
	PRIMARY KEY (id)
)

;

CREATE TABLE locations (
	id VARCHAR NOT NULL,
	name VARCHAR NOT NULL,
	PRIMARY KEY (id)
)

;

CREATE TABLE review_timings (
	id VARCHAR NOT NULL,
	task VARCHAR NOT NULL,
	mode VARCHAR NOT NULL,
	seconds FLOAT NOT NULL,
	PRIMARY KEY (id)
)

;

CREATE TABLE cameras (
	id VARCHAR NOT NULL,
	location_id VARCHAR NOT NULL,
	name VARCHAR NOT NULL,
	connections JSON NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(location_id) REFERENCES locations (id)
)

;
CREATE INDEX ix_cameras_location_id ON cameras (location_id);

CREATE TABLE entities (
	id VARCHAR NOT NULL,
	camera_id VARCHAR NOT NULL,
	object_type VARCHAR NOT NULL,
	attributes JSON NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(camera_id) REFERENCES cameras (id)
)

;
CREATE INDEX ix_entities_camera_id ON entities (camera_id);
CREATE INDEX ix_entities_object_type ON entities (object_type);

CREATE TABLE videos (
	id VARCHAR NOT NULL,
	camera_id VARCHAR NOT NULL,
	filename VARCHAR NOT NULL,
	path VARCHAR NOT NULL,
	recording_start VARCHAR NOT NULL,
	duration FLOAT NOT NULL,
	fps FLOAT NOT NULL,
	frame_count INTEGER NOT NULL,
	width INTEGER NOT NULL,
	height INTEGER NOT NULL,
	sha256 VARCHAR NOT NULL,
	status VARCHAR NOT NULL,
	progress FLOAT NOT NULL,
	error TEXT,
	is_demo BOOLEAN NOT NULL,
	pipeline VARCHAR NOT NULL,
	created_at VARCHAR NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(camera_id) REFERENCES cameras (id)
)

;
CREATE INDEX ix_videos_status ON videos (status);
CREATE INDEX ix_videos_camera_id ON videos (camera_id);
CREATE INDEX ix_videos_recording_start ON videos (recording_start);

CREATE TABLE tracks (
	id VARCHAR NOT NULL,
	video_id VARCHAR NOT NULL,
	entity_id VARCHAR NOT NULL,
	tracker VARCHAR NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(video_id) REFERENCES videos (id),
	FOREIGN KEY(entity_id) REFERENCES entities (id)
)

;
CREATE INDEX ix_tracks_video_id ON tracks (video_id);
CREATE INDEX ix_tracks_entity_id ON tracks (entity_id);

CREATE TABLE observations (
	id VARCHAR NOT NULL,
	video_id VARCHAR NOT NULL,
	entity_id VARCHAR NOT NULL,
	track_id VARCHAR NOT NULL,
	start FLOAT NOT NULL,
	"end" FLOAT NOT NULL,
	boxes JSON NOT NULL,
	confidence FLOAT,
	attributes JSON NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(video_id) REFERENCES videos (id),
	FOREIGN KEY(entity_id) REFERENCES entities (id),
	FOREIGN KEY(track_id) REFERENCES tracks (id)
)

;
CREATE INDEX ix_observations_entity_id ON observations (entity_id);
CREATE INDEX ix_observations_track_id ON observations (track_id);
CREATE INDEX ix_observations_video_id ON observations (video_id);

CREATE TABLE embeddings (
	id VARCHAR NOT NULL,
	observation_id VARCHAR NOT NULL,
	model VARCHAR NOT NULL,
	values JSON NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(observation_id) REFERENCES observations (id)
)

;
CREATE INDEX ix_embeddings_observation_id ON embeddings (observation_id);
CREATE INDEX ix_embeddings_model ON embeddings (model);

CREATE TABLE evidence (
	id VARCHAR NOT NULL,
	video_id VARCHAR NOT NULL,
	observation_id VARCHAR,
	frame_start INTEGER NOT NULL,
	frame_end INTEGER NOT NULL,
	timestamp_start FLOAT NOT NULL,
	timestamp_end FLOAT NOT NULL,
	sha256 VARCHAR NOT NULL,
	method VARCHAR NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(video_id) REFERENCES videos (id),
	FOREIGN KEY(observation_id) REFERENCES observations (id)
)

;
CREATE INDEX ix_evidence_observation_id ON evidence (observation_id);
CREATE INDEX ix_evidence_video_id ON evidence (video_id);

CREATE TABLE embedding_vectors (
	embedding_id VARCHAR NOT NULL,
	model VARCHAR NOT NULL,
	vector VECTOR NOT NULL,
	PRIMARY KEY (embedding_id),
	FOREIGN KEY(embedding_id) REFERENCES embeddings (id)
)

;
CREATE INDEX ix_embedding_vectors_model ON embedding_vectors (model);

CREATE TABLE events (
	id VARCHAR NOT NULL,
	evidence_id VARCHAR NOT NULL,
	camera_id VARCHAR NOT NULL,
	event_type VARCHAR NOT NULL,
	title VARCHAR NOT NULL,
	description TEXT NOT NULL,
	start VARCHAR NOT NULL,
	"end" VARCHAR NOT NULL,
	confidence FLOAT,
	category VARCHAR NOT NULL,
	attributes JSON NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(evidence_id) REFERENCES evidence (id),
	FOREIGN KEY(camera_id) REFERENCES cameras (id)
)

;
CREATE INDEX ix_events_event_type ON events (event_type);
CREATE INDEX ix_events_camera_id ON events (camera_id);
CREATE INDEX ix_events_evidence_id ON events (evidence_id);
CREATE INDEX ix_events_start ON events (start);

CREATE TABLE relationships (
	id VARCHAR NOT NULL,
	source_id VARCHAR NOT NULL,
	target_id VARCHAR NOT NULL,
	evidence_id VARCHAR NOT NULL,
	relation VARCHAR NOT NULL,
	category VARCHAR NOT NULL,
	signals JSON NOT NULL,
	explanation TEXT NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(source_id) REFERENCES entities (id),
	FOREIGN KEY(target_id) REFERENCES entities (id),
	FOREIGN KEY(evidence_id) REFERENCES evidence (id)
)

;
CREATE INDEX ix_relationships_source_id ON relationships (source_id);
CREATE INDEX ix_relationships_evidence_id ON relationships (evidence_id);
CREATE INDEX ix_relationships_target_id ON relationships (target_id);

CREATE TABLE benchmark_ground_truth (
	query_id VARCHAR NOT NULL,
	event_id VARCHAR NOT NULL,
	PRIMARY KEY (query_id, event_id),
	FOREIGN KEY(query_id) REFERENCES benchmark_queries (id),
	FOREIGN KEY(event_id) REFERENCES events (id)
)

;

CREATE TABLE event_entities (
	event_id VARCHAR NOT NULL,
	entity_id VARCHAR NOT NULL,
	PRIMARY KEY (event_id, entity_id),
	FOREIGN KEY(event_id) REFERENCES events (id),
	FOREIGN KEY(entity_id) REFERENCES entities (id)
)

;

CREATE TABLE investigation_findings (
	investigation_id VARCHAR NOT NULL,
	event_id VARCHAR NOT NULL,
	PRIMARY KEY (investigation_id, event_id),
	FOREIGN KEY(investigation_id) REFERENCES investigations (id),
	FOREIGN KEY(event_id) REFERENCES events (id)
)

;
