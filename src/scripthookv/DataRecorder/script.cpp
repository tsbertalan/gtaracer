#include "script.h"
#include "keyboard.h"
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>


#define DEBUGMODE FALSE
#define SCAN_DIST 64.0

double get_wall_time() {
	using namespace std::chrono;
	steady_clock::time_point now = std::chrono::steady_clock::now();
	auto us = now.time_since_epoch() / std::chrono::microseconds(1);
	return us / 1e6;
}


typedef struct EntityState {
	const char m1 = 'G';
	const char m2 = 'T';
	const char m3 = 'A';
	const char m4 = 'G';
	const char m5 = 'T';
	const char m6 = 'A';
	int id;
	float
		posx, posy, posz,
		roll, pitch, yaw,
		velx, vely, velz,
		rvelx, rvely, rvelz;
	double wall_time;
	float screenx, screeny;
	bool occluded;
	bool is_vehicle;
	bool is_player;
} EntityState;


char entityStateChecksum(EntityState& es) {
	char *bytes = reinterpret_cast<char*>(&es);
	char csum = 0;
	for (uint i = 0; i < sizeof(es) - 1; i++) {
		csum ^= bytes[i];
	}
	return csum;
}


class BinaryWriter {
private:
	std::ofstream fh;
	std::ofstream& log;
	bool file_opened;
public:
	BinaryWriter(std::ofstream& log_file) : log(log_file) {
		file_opened = FALSE;

	}

	std::ostringstream generate_filename() {
		std::ostringstream new_name;
		new_name << "GTA_recording-" << get_wall_time() << ".bin";
		return new_name;
	}

	void open_file(const char *file_name = nullptr) {
		if (!file_name) {
			// https://stackoverflow.com/questions/1374468/stringstream-string-and-char-conversion-confusion
			const std::string& tmp = generate_filename().str();
			file_name = tmp.c_str();
		}

		if (!file_opened) {
			log << "Binary data will be saved to file \"" << file_name << "\"." << std::endl;
			fh.open(file_name, std::ios::out | std::ios::trunc | std::ios::binary);
			file_opened = TRUE;
		}
	}

	void close_file() {
		if (file_opened) {
			log << "Closing open binary file." << std::endl;
			flush();
			file_opened = FALSE;
			fh.close();
		}
	}

	bool saveData(EntityState& data) {
		if (file_opened) {
			fh.write(reinterpret_cast<char*>(&data), sizeof(data));
			fh << entityStateChecksum(data);
			return TRUE;
		}
		else {
			log << "File is not open; refusing to save new data." << std::endl;
			return FALSE;
		}

	}

	void flush() {
		if (file_opened) {
			fh << std::flush;
		}
	}

};


void drawText(char *text, float x, float y, float box_height = 0, float box_width = 0) {
	UI::SET_TEXT_FONT(eFont::FontChaletComprimeCologne);
	UI::SET_TEXT_SCALE(0.2, 0.2);
	UI::SET_TEXT_COLOUR(200, 255, 200, 255);
	UI::SET_TEXT_WRAP(0.0, 1.0); // Wrap to screen left and right.
	UI::SET_TEXT_CENTRE(0);
	UI::SET_TEXT_DROPSHADOW(0, 0, 0, 0, 0);
	UI::SET_TEXT_EDGE(1, 0, 0, 0, 205);
	UI::_SET_TEXT_ENTRY("STRING");
	UI::_ADD_TEXT_COMPONENT_STRING(text);
	UI::_DRAW_TEXT(x, y);
	// box
	if (box_height > 0 && box_width > 0)
		GRAPHICS::DRAW_RECT(x + 0.027f, y + 0.043f, box_width, box_height, 75, 75, 75, 75);
}


bool drawTextOnObject(char *text, Entity entity, float box_height = 0.058f, float box_width = 0.08f) {
	Vector3 v = ENTITY::GET_ENTITY_COORDS(entity, TRUE);
	float x, y;
	if (GRAPHICS::_WORLD3D_TO_SCREEN2D(v.x, v.y, v.z, &x, &y))
	{
		drawText(text, x, y);
		return true;
	}
	else {
		return false;
	}
}


std::ostringstream formatEntityState(Entity entity) {
	Vector3 pos = ENTITY::GET_ENTITY_COORDS(entity, TRUE);
	//float roll = ENTITY::GET_ENTITY_ROLL(entity);
	//float pitch = ENTITY::GET_ENTITY_PITCH(entity);
	//float yaw = ENTITY::GET_ENTITY_HEADING(entity);
	//Vector3 vel = ENTITY::GET_ENTITY_VELOCITY(entity);
	//Vector3 rvel = ENTITY::GET_ENTITY_ROTATION_VELOCITY(entity);

	bool occluded = ENTITY::IS_ENTITY_OCCLUDED(entity);

	float xvis, yvis;
	if (!GRAPHICS::_WORLD3D_TO_SCREEN2D(pos.x, pos.y, pos.z, &xvis, &yvis)) {
		xvis = -1;
		yvis = -1;
	}

	std::ostringstream text;
	text << std::fixed << std::setprecision(1);
	//text << "Pos (" << pos.x << "," << pos.y << "," << pos.z << ")" << std::endl << std::flush;
	//text << "Rot (" << roll << "," << pitch << "," << yaw << ")" << std::endl << std::flush;
	//text << "PosVel (" << vel.x << "," << vel.y << "," << vel.z << ")" << std::endl << std::flush;
	//text << "RotVel (" << rvel.x << "," << rvel.y << "," << rvel.z << ")" << std::endl << std::flush;
	text << "Screen (" << xvis << "," << yvis << ")" << std::endl << std::flush;
	if (occluded) {
		text << "occluded";
	}
	else {
		text << "not occluded";
	}

	return text;
}


EntityState examineEntity(double wall_time, Entity entity, Ped player_ped, Vehicle player_vehicle) {
	EntityState s;
	s.wall_time = wall_time;

	s.occluded = ENTITY::IS_ENTITY_OCCLUDED(entity);

	s.roll = ENTITY::GET_ENTITY_ROLL(entity);
	s.pitch = ENTITY::GET_ENTITY_PITCH(entity);
	s.yaw = ENTITY::GET_ENTITY_HEADING(entity);

	Vector3 p = ENTITY::GET_ENTITY_COORDS(entity, TRUE);
	s.posx = p.x;
	s.posy = p.y;
	s.posz = p.z;


	Vector3 v = ENTITY::GET_ENTITY_VELOCITY(entity);
	s.velx = v.x;
	s.vely = v.y;
	s.velz = v.z;

	Vector3 rvel = ENTITY::GET_ENTITY_ROTATION_VELOCITY(entity);
	s.rvelx = rvel.x;
	s.rvely = rvel.y;
	s.rvelz = rvel.z;

	float xvis = -1;
	float yvis = -1;
	if (!GRAPHICS::_WORLD3D_TO_SCREEN2D(p.x, p.y, p.z, &xvis, &yvis)) {
		xvis = -1;
		yvis = -1;
	}
	s.screenx = xvis;
	s.screeny = yvis;

	s.is_vehicle = ENTITY::IS_ENTITY_A_VEHICLE(entity);
	if (s.is_vehicle) {
		s.id = ENTITY::GET_VEHICLE_INDEX_FROM_ENTITY_INDEX(entity);
		s.is_player = player_vehicle == s.id;
	}
	else {
		s.id = ENTITY::GET_PED_INDEX_FROM_ENTITY_INDEX(entity);
		s.is_player = player_ped == s.id;
	}


	//s.dist_to_player = GAMEPLAY::GET_DISTANCE_BETWEEN_COORDS(
	//	player_coords.x, player_coords.y, player_coords.z,
	//	p.x, p.y, p.z,
	//	TRUE);

	return s;
}


void update(BinaryWriter& binary_writer, std::ofstream& log, bool& is_currently_recording, bool& first_keyhit_seen, bool& writing_lock_set)
{

	if (writing_lock_set)
		// Don't do anything if the lock is set.
		return;

	if (IsKeyDown(VK_NUMPAD1)) {
		if (!first_keyhit_seen) { // Only detect actual down changes; don't repeat while down.
			if (is_currently_recording) {
				// If we're currently recording, stop.
				binary_writer.close_file();
			}
			else {
				binary_writer.open_file();
			}
			is_currently_recording = !is_currently_recording;
			first_keyhit_seen = TRUE;
		}
	}
	else {
		first_keyhit_seen = FALSE;
	}


	if (is_currently_recording) {

		// TODO: Get more important fields : (1) vehicle type(for example, so we can ignore airplanes easily) (2) ? ? ? (3) profit.
		// TODO: Use GET_MODEL_DIMENSIONS to get the bounding box for the entity.
		// TODO: Use some of the CAM namespace actions to get scene information once per update. (E.g., camera FoV and position.)

		double wall_time;


		// we don't want to mess with missions in this example
		if (GAMEPLAY::GET_MISSION_FLAG())
			return;

		// If we've gotten this far, we're committed to writing *something*.
		writing_lock_set = true;


		// Get the player.
		Player player = PLAYER::PLAYER_ID();
		Ped player_ped = PLAYER::PLAYER_PED_ID();
		Vehicle last_player_vehicle = PLAYER::GET_PLAYERS_LAST_VEHICLE();
		Vector3 plv = ENTITY::GET_ENTITY_COORDS(player_ped, TRUE);



		// Check if player ped exists and control is on (e.g. not in a cutscene).
		if (!ENTITY::DOES_ENTITY_EXIST(player_ped) || !PLAYER::IS_PLAYER_CONTROL_ON(player))
			return;

		wall_time = get_wall_time();
		binary_writer.saveData(examineEntity(wall_time, last_player_vehicle, player_ped, last_player_vehicle));


		// Get all vehicles.
		const int ARR_SIZE = 1024;
		Vehicle vehicles[ARR_SIZE];
		wall_time = get_wall_time();
		int count = worldGetAllVehicles(vehicles, ARR_SIZE);

		for (int i = 0; i < count; i++)
		{
			// This model could be useful to put in the packet--"object kind/type"?
			// There's a mapping (GET_DISPLAY_NAME_FROM_VEHICLE_MODEL) from these to char* names;
			// we can reproduce that in Python as needed.
			Hash model = ENTITY::GET_ENTITY_MODEL(vehicles[i]);

			Vector3 v = ENTITY::GET_ENTITY_COORDS(vehicles[i], TRUE);

			float dist = GAMEPLAY::GET_DISTANCE_BETWEEN_COORDS(plv.x, plv.y, plv.z, v.x, v.y, v.z, TRUE);
			if (dist < SCAN_DIST) {
				binary_writer.saveData(examineEntity(wall_time, vehicles[i], player_ped, last_player_vehicle));

				if (DEBUGMODE) {
					float x, y;
					if (GRAPHICS::_WORLD3D_TO_SCREEN2D(v.x, v.y, v.z, &x, &y))
					{
						int health = ENTITY::GET_ENTITY_HEALTH(vehicles[i]);
						char *name = VEHICLE::GET_DISPLAY_NAME_FROM_VEHICLE_MODEL(model);
						// print text in a box
						char text[2048];
						std::ostringstream state_string = formatEntityState(vehicles[i]);
						sprintf_s(
							text,
							"%s\n%s\nDist %.1f",
							name,
							state_string.str().c_str(),
							dist
						);
						drawTextOnObject(text, vehicles[i]);
					}
				}
			}

		}


		// Get all peds.
		Ped peds[ARR_SIZE];
		wall_time = get_wall_time();
		count = worldGetAllPeds(peds, ARR_SIZE);

		for (int i = 0; i < count; i++) {

			Vector3 v = ENTITY::GET_ENTITY_COORDS(peds[i], TRUE);
			float dist = GAMEPLAY::GET_DISTANCE_BETWEEN_COORDS(plv.x, plv.y, plv.z, v.x, v.y, v.z, TRUE);
			bool in_car = PED::IS_PED_IN_ANY_VEHICLE(peds[i], FALSE);
			if (dist < SCAN_DIST && !in_car) {

				binary_writer.saveData(examineEntity(wall_time, peds[i], player_ped, last_player_vehicle));

				if (DEBUGMODE) {
					char text[2048];
					int ped_type = PED::GET_PED_TYPE(peds[i]);
					std::ostringstream state_string = formatEntityState(peds[i]);
					sprintf_s(text, "Ped #%d (type %d)\n%s", i, ped_type, state_string.str().c_str());
					drawTextOnObject(text, peds[i], 0.025f);
				}
			}
		}

		// Once we're done, we clear the lock.
		writing_lock_set = false;
	}
}


void main()
{
	std::ofstream log;
	log.open("GTA_recording.log", std::ios_base::app);

	BinaryWriter binary_writer(log);

	bool writing_lock_set = false;

	if (DEBUGMODE)
		log << "Each struct will have size " << sizeof(EntityState) << ", followed by a checksum byte." << std::endl;

	bool recording_state = FALSE;
	bool key_toggling = FALSE;

	while (true)
	{
		update(binary_writer, log, recording_state, key_toggling, writing_lock_set);
		binary_writer.flush();
		WAIT(0);
	}

	log.close();
}


void ScriptMain()
{
	srand(GetTickCount());
	main();
}
