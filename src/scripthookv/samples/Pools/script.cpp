#include "script.h"
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>


#define DEBUGMODE FALSE


double get_wall_time() {
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq)) {
		//  Handle error
		return 0;
	}
	if (!QueryPerformanceCounter(&time)) {
		//  Handle error
		return 0;
	}
	return (double)time.QuadPart / freq.QuadPart;
}


typedef struct EntityState {
	const char m1 = 'G';
	const char m2 = 'T';
	const char m3 = 'A';
	uint id;  // 4
	float // 4 * 12
		posx, posy, posz,
		roll, pitch, yaw,
		velx, vely, velz,
		rvelx, rvely, rvelz;
	double wall_time; // 8
	float screenx, screeny; // 4*2
	bool occluded; // 1
	bool is_vehicle; // 1
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
public:
	BinaryWriter(char *file_name, std::ofstream& log_file) : log(log_file) {
		log << "Binary data will be saved to file \"" << file_name << "\"." << std::endl;
		fh.open(file_name, std::ios::trunc);
	}

	bool saveData(EntityState& data) {
		fh.write(reinterpret_cast<char*>(&data), sizeof(data) - 1);
		fh << entityStateChecksum(data);
		return TRUE;
	}

};


void drawText(char *text, float x, float y, float box_height=0, float box_width=0) {
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
	if(box_height > 0 && box_width > 0)
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


EntityState examineEntity(double wall_time, int id, Entity entity) {
	EntityState s;
	s.id = id;
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
	
	float xvis, yvis;
	if (!GRAPHICS::_WORLD3D_TO_SCREEN2D(p.x, p.y, p.z, &xvis, &yvis)) {
		xvis = -1;
		yvis = -1;
	}
	s.screenx = xvis;
	s.screeny = yvis;

	s.is_vehicle = ENTITY::IS_ENTITY_A_VEHICLE(entity);

	return s;
}



void update(BinaryWriter& binary_writer, std::ofstream& log_file)
{
	double wall_time;


	// we don't want to mess with missions in this example
	if (GAMEPLAY::GET_MISSION_FLAG())
		return;

	
	// Get the player.
	Player player = PLAYER::PLAYER_ID();
	Ped playerPed = PLAYER::PLAYER_PED_ID();


	// Check if player ped exists and control is on (e.g. not in a cutscene).
	if (!ENTITY::DOES_ENTITY_EXIST(playerPed) || !PLAYER::IS_PLAYER_CONTROL_ON(player))
		return;


	// Get all vehicles.
	const int ARR_SIZE = 1024;
	Vehicle vehicles[ARR_SIZE];
	int count = worldGetAllVehicles(vehicles, ARR_SIZE);
	wall_time = get_wall_time();
	
	Vector3 plv = ENTITY::GET_ENTITY_COORDS(playerPed, TRUE);
	for (int i = 0; i < count; i++)
	{
		//binary_writer.saveData(examineEntity(wall_time , i, vehicles[i]));

		// This model could be useful to put in the packet--"object kind/type"?
		// There's a mapping (GET_DISPLAY_NAME_FROM_VEHICLE_MODEL) from these to char* names;
		// we can reproduce that in Python as needed.
		Hash model = ENTITY::GET_ENTITY_MODEL(vehicles[i]);

		Vector3 v = ENTITY::GET_ENTITY_COORDS(vehicles[i], TRUE);
		if (DEBUGMODE) {
			float x, y;
			if (GRAPHICS::_WORLD3D_TO_SCREEN2D(v.x, v.y, v.z, &x, &y))
			{
				float dist = GAMEPLAY::GET_DISTANCE_BETWEEN_COORDS(plv.x, plv.y, plv.z, v.x, v.y, v.z, TRUE);
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


	// Get all peds.
	Ped peds[ARR_SIZE];
	count = worldGetAllPeds(peds, ARR_SIZE);
	wall_time = get_wall_time();
	//log_file << "Found " << count << " peds total." << std::endl << std::flush;

	for (int i = 0; i < count; i++) {
		//client.sendData(examineEntity(wall_time, i, peds[i]));
		binary_writer.saveData(examineEntity(wall_time, i, peds[i]));
		char text[2048];
		int ped_type = PED::GET_PED_TYPE(peds[i]);
		if (DEBUGMODE) {
			std::ostringstream state_string = formatEntityState(peds[i]);
			sprintf_s(text, "Ped #%d (type %d)\n%s", i, ped_type, state_string.str().c_str());
			drawTextOnObject(text, peds[i], 0.025f);
		}
	}
}


void main()
{		
	std::ofstream log_file;
	log_file.open("GTA_recording.log", std::ios_base::app);

	BinaryWriter binary_writer("GTA_recording.bin", log_file);

	log_file << "Each struct will have size " << sizeof(EntityState) << "." << std::endl;

	while (true)
	{
	    update(binary_writer, log_file);
		WAIT(0);
	}

	log_file.close();
}


void ScriptMain()
{	
	srand(GetTickCount());
	main();
}