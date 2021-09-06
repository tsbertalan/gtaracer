/*
	THIS FILE IS A PART OF GTA V SCRIPT HOOK SDK
				http://dev-c.com			
			(C) Alexander Blade 2015
*/

// https://www.zachburlingame.com/2011/05/resolving-redefinition-errors-betwen-ws2def-h-and-winsock-h/
#define WIN32_LEAN_AND_MEAN


#include "script.h"
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>



//#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h> // getaddrinfo, wsaData, hints
//#include <stdlib.h>
//#include <stdio.h>


// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
// Who knows why; these Win32 "tutorials" don't explain anything.
// https://docs.microsoft.com/en-us/windows/win32/winsock/complete-client-code
// This does remove the cryptic "error LNK2001: unresolved external symbol __imp_WSAStartup", though (see https://stackoverflow.com/questions/16948064/unresolved-external-symbol-lnk2019).
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")


#define DEFAULT_BUFLEN 512
#define DEFAULT_PORT "27015"

DWORD	vehUpdateTime;
DWORD	pedUpdateTime;

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
		fh.open(file_name, std::ios::trunc);
	}

	bool saveData(EntityState& data) {
		//log << "Save data to file handle ..." << std::endl << std::flush;c++ seri
		//fh.write("GTADATAMARKER", 13);
		fh.write(reinterpret_cast<char*>(&data), sizeof(data) - 1);
		fh << entityStateChecksum(data);
		//log << "saved " << sizeof(data) << " data bytes." << std::endl << std::flush;
		//log << "data.posx=" << data.posx << std::endl;
		return TRUE;
	}

};



class Client {
private:
	int iResult;
	SOCKET ConnectSocket;
	char recvbuf[DEFAULT_BUFLEN];
	int recvbuflen = DEFAULT_BUFLEN;
	std::ofstream& log;
	

public:
	bool initialized;
	Client(std::ofstream& log_file) : log(log_file), initialized(FALSE) {}
	bool setUp() {
		log << "Setting up UDP client ..." << std::endl << std::flush;

		WSADATA wsaData;
		ConnectSocket = INVALID_SOCKET;
		struct addrinfo *result = NULL,
			*ptr = NULL,
			hints;
		

		// Initialize Winsock
		iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
		if (iResult != 0) {
			log << "WSAStartup failed with error: " << iResult << std::endl << std::flush;
			return false;
		}

		ZeroMemory(&hints, sizeof(hints));
		hints.ai_family = AF_UNSPEC;
		hints.ai_socktype = SOCK_STREAM;
		hints.ai_protocol = IPPROTO_TCP;

		// Resolve the server address and port.
		//log << "Resolve the server address and port." << std::endl << std::flush;
		char *addr = "localhost";
		iResult = getaddrinfo(addr, DEFAULT_PORT, &hints, &result);
		if (iResult != 0) {
			log << "getaddrinfo failed with error: " << iResult << std::endl << std::flush;
			WSACleanup();
			return FALSE;
		}

		// Attempt to connect to an address until one succeeds.
		for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {

			// Create a SOCKET for connecting to server.
			//log << "Create a SOCKET for connecting to server." << std::endl << std::flush;
			ConnectSocket = socket(ptr->ai_family, ptr->ai_socktype,
				ptr->ai_protocol);
			if (ConnectSocket == INVALID_SOCKET) {
				log << "socket failed with error: " << WSAGetLastError() << std::endl << std::flush;
				WSACleanup();
				return FALSE;
			}

			// Connect to server.
			//log << "Connect to server." << std::endl << std::flush;
			iResult = connect(ConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
			if (iResult == SOCKET_ERROR) {
				closesocket(ConnectSocket);
				ConnectSocket = INVALID_SOCKET;
				continue;
			}
			break;
		}


		freeaddrinfo(result);

		if (ConnectSocket == INVALID_SOCKET) {
			log << "unable to connect to UPD server." << std::endl << std::flush;
			WSACleanup();
			return FALSE;
		}

		log << "connected to UDP server!" << std::endl << std::flush;

		initialized = TRUE;

		return TRUE;

	}

	bool sendData(EntityState data) {
		log << "Send data to UDP server ..." << std::endl << std::flush;
		iResult = send(ConnectSocket, reinterpret_cast<char*>(&data), sizeof(data), 0);
		if (iResult == SOCKET_ERROR) {
			log << "send failed with error: " << WSAGetLastError() << std::endl << std::flush;
			//closesocket(ConnectSocket);
			//WSACleanup();
			return FALSE;
		}

		log << "bytes Sent: " << iResult << std::endl << std::flush;
		return TRUE;
	}


	//int recvData() {
	//	// Receive until the peer closes the connection
	//	do {
	//		iResult = recv(ConnectSocket, recvbuf, recvbuflen, 0);
	//		if (iResult > 0)
	//			printf("Bytes received: %d\n", iResult);
	//		else if (iResult == 0)
	//			printf("Connection closed\n");
	//		else {
	//			printf("recv failed with error: %d\n", WSAGetLastError());
	//			return -1;
	//		}
	//	} while (iResult > 0);
	//
	//	return iResult;
	//}


	bool cleanUp(){
		// Shutdown the connection since no more data will be sent.
		log << "Shutdown the connection since no more data will be sent." << std::endl << std::flush;
		iResult = shutdown(ConnectSocket, SD_SEND);
		if (iResult == SOCKET_ERROR) {
			printf("shutdown failed with error: %d\n", WSAGetLastError());
			closesocket(ConnectSocket);
			WSACleanup();
			return FALSE;
		}

		// cleanup
		log << "Clean up the UDP client." << std::endl << std::flush;
		closesocket(ConnectSocket);
		WSACleanup();
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


std::ostringstream formatEntityState(Entity entity, bool pretty = TRUE) {
	Vector3 pos = ENTITY::GET_ENTITY_COORDS(entity, TRUE);
	float roll = ENTITY::GET_ENTITY_ROLL(entity);
	float pitch = ENTITY::GET_ENTITY_PITCH(entity);
	float yaw = ENTITY::GET_ENTITY_HEADING(entity);
	Vector3 vel = ENTITY::GET_ENTITY_VELOCITY(entity);
	Vector3 rvel = ENTITY::GET_ENTITY_ROTATION_VELOCITY(entity);

	bool occluded = ENTITY::IS_ENTITY_OCCLUDED(entity);

	std::ostringstream visible;
	float xvis, yvis;
	if (!GRAPHICS::_WORLD3D_TO_SCREEN2D(pos.x, pos.y, pos.z, &xvis, &yvis)) {
		xvis = -1;
		yvis = -1;
	}

	std::ostringstream text;
	if (pretty) {
		text << std::fixed << std::setprecision(1);
		//text << "Pos (" << pos.x << "," << pos.y << "," << pos.z << ")" << std::endl << std::flush;
		//text << "Rot (" << roll << "," << pitch << "," << yaw << ")" << std::endl << std::flush;
		text << "PosVel (" << vel.x << "," << vel.y << "," << vel.z << ")" << std::endl << std::flush;
		//text << "RotVel (" << rvel.x << "," << rvel.y << "," << rvel.z << ")" << std::endl << std::flush;
		/*text << "Screen (" << xvis << "," << yvis << ")" << std::endl << std::flush;
		if (occluded) {
			text << "occluded";
		}
		else {
			text << "not occluded";
		}*/
		
	}
	else {
		text << pos.x << "," << pos.y << "," << pos.z << "," << std::endl << std::flush;
		text << roll << "," << pitch << "," << yaw << "," << std::endl << std::flush;
		text << vel.x << "," << vel.y << "," << vel.z << "," << std::endl << std::flush;
		text << rvel.x << "," << rvel.y << "," << rvel.z << "," << std::endl << std::flush;
		text << xvis << "," << yvis << "," << std::endl << std::flush;
		text << occluded;
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

	Player player = PLAYER::PLAYER_ID();
	Ped playerPed = PLAYER::PLAYER_PED_ID();

	// check if player ped exists and control is on (e.g. not in a cutscene)
	if (!ENTITY::DOES_ENTITY_EXIST(playerPed) || !PLAYER::IS_PLAYER_CONTROL_ON(player))
		return;

	// get all vehicles
	const int ARR_SIZE = 1024;
	Vehicle vehicles[ARR_SIZE];
	int count = worldGetAllVehicles(vehicles, ARR_SIZE);
	wall_time = get_wall_time();
	//log_file << "Found " << count << " vehicles total." << std::endl << std::flush;

	//// randomize all vehicle colours every 200 ms
	//// setting only primary or secondary color per update
	//if (vehUpdateTime + 200 < GetTickCount())
	//{
	//	vehUpdateTime = GetTickCount();
	//	for (int i = 0; i < count; i++)
	//	{
	//		int primary = 0, secondary = 0;
	//		VEHICLE::GET_VEHICLE_COLOURS(vehicles[i], &primary, &secondary);
	//		if (rand() % 2)
	//			VEHICLE::SET_VEHICLE_COLOURS(vehicles[i], rand() % (VehicleColorBrushedGold + 1), secondary);
	//		else
	//			VEHICLE::SET_VEHICLE_COLOURS(vehicles[i], primary, rand() % (VehicleColorBrushedGold + 1));
	//	}		
	//}

	/*	
	// delete all vehicles
	for (int i = 0; i < count; i++)
	{		
		if (!ENTITY::IS_ENTITY_A_MISSION_ENTITY(vehicles[i]))
			ENTITY::SET_ENTITY_AS_MISSION_ENTITY(vehicles[i], TRUE, TRUE);
		VEHICLE::DELETE_VEHICLE(&vehicles[i]);
	}
	*/

	
	

	// let's track all exising vehicles
	Vector3 plv = ENTITY::GET_ENTITY_COORDS(playerPed, TRUE);
	for (int i = 0; i < count; i++)
	{
		//log_file << "UDP client initialized? " << client.initialized << std::endl;
		//if (client.initialized)
			//client.sendData(examineEntity(i, vehicles[i]));
		//binary_writer.saveData(examineEntity(wall_time , i, vehicles[i]));

		Hash model = ENTITY::GET_ENTITY_MODEL(vehicles[i]);
		//if (VEHICLE::IS_THIS_MODEL_A_HELI(model) || VEHICLE::IS_THIS_MODEL_A_PLANE(model) || ENTITY::IS_ENTITY_IN_AIR(vehicles[i]))
		//{
			Vector3 v = ENTITY::GET_ENTITY_COORDS(vehicles[i], TRUE);
			float x, y;
			if (GRAPHICS::_WORLD3D_TO_SCREEN2D(v.x, v.y, v.z, &x, &y))
			{
				float dist = GAMEPLAY::GET_DISTANCE_BETWEEN_COORDS(plv.x, plv.y, plv.z, v.x, v.y, v.z, TRUE);
				// draw text if vehicle isn't close to the player
				//if (dist > 15.0)
				//{
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
					//drawText(text, x, y);
					drawTextOnObject(text, vehicles[i]);
				//}
			}
		//}
	}

	// get all peds
	Ped peds[ARR_SIZE];
	count = worldGetAllPeds(peds, ARR_SIZE);
	wall_time = get_wall_time();
	//log_file << "Found " << count << " peds total." << std::endl << std::flush;

	for (int i = 0; i < count; i++) {
		//client.sendData(examineEntity(wall_time, i, peds[i]));
		binary_writer.saveData(examineEntity(wall_time, i, peds[i]));
		char text[2048];
		int ped_type = PED::GET_PED_TYPE(peds[i]);
		std::ostringstream state_string = formatEntityState(peds[i]);
		sprintf_s(text, "Ped #%d (type %d)\n%s", i, ped_type, state_string.str().c_str());
		drawTextOnObject(text, peds[i], 0.025f);
	}

	//// randmoize peds
	//if (pedUpdateTime + 200 < GetTickCount())
	//{
	//	pedUpdateTime = GetTickCount();
	//	for (int i = 0; i < count; i++)
	//	{
	//		// if (rand() % 2 != 0) continue;
	//		if (peds[i] != playerPed && PED::IS_PED_HUMAN(peds[i]) && !ENTITY::IS_ENTITY_DEAD(peds[i]))
	//		{
	//			for (int component = 0; component < 12; component++)
	//			{
	//				if (rand() % 2 != 0) continue;
	//				for (int j = 0; j < 100; j++)
	//				{
	//					int drawable = rand() % 10;
	//					int texture = rand() % 10;
	//					if (PED::IS_PED_COMPONENT_VARIATION_VALID(peds[i], component, drawable, texture))
	//					{
	//						PED::SET_PED_COMPONENT_VARIATION(peds[i], component, drawable, texture, 0);
	//						break;
	//					}
	//				}
	//			}
	//		}
	//	}
	//}
		
	// get all objects
	//Object objects[ARR_SIZE];
	//count = worldGetAllObjects(objects, ARR_SIZE);

	// mark objects on the screen around the player

	// there are lots of objects in some places so we need to
	// remove possibilty of text being drawn on top of another text
	// thats why we will check distance between text on screen
	//std::vector<std::pair<float, float>> coordsOnScreen;
	//for (int i = 0; i < count; i++)
	//{
	//	Hash model = ENTITY::GET_ENTITY_MODEL(objects[i]);
	//	Vector3 v = ENTITY::GET_ENTITY_COORDS(objects[i], TRUE);
	//	float x, y;
	//	if (GRAPHICS::_WORLD3D_TO_SCREEN2D(v.x, v.y, v.z, &x, &y))
	//	{
	//		// select objects only around
	//		Vector3 plv = ENTITY::GET_ENTITY_COORDS(playerPed, TRUE);
	//		float dist = GAMEPLAY::GET_DISTANCE_BETWEEN_COORDS(plv.x, plv.y, plv.z, v.x, v.y, v.z, TRUE);
	//		if (dist < 200.0)
	//		{
	//			// check if the text fits on screen
	//			bool bFitsOnscreen = true;
	//			for (auto iter = coordsOnScreen.begin(); iter != coordsOnScreen.end(); iter++)
	//			{
	//				float textDist = sqrtf((iter->first - x)*(iter->first - x) + (iter->second - y)*(iter->second - y));
	//				if (textDist < 0.05)
	//				{
	//					bFitsOnscreen = false;
	//					break;
	//				}
	//			}
	//			// if text doesn't fit then skip draw
	//			if (!bFitsOnscreen) continue;

	//			// add text coords to the vector
	//			coordsOnScreen.push_back({ x, y });

	//			// print text in a box
	//			char text[256];
	//			sprintf_s(text, "^\n%08X\n%.02f", model, dist);
	//			UI::SET_TEXT_FONT(0);
	//			UI::SET_TEXT_SCALE(0.2, 0.2);
	//			UI::SET_TEXT_COLOUR(200, 255, 200, 255);
	//			UI::SET_TEXT_WRAP(0.0, 1.0);
	//			UI::SET_TEXT_CENTRE(0);
	//			UI::SET_TEXT_DROPSHADOW(0, 0, 0, 0, 0);
	//			UI::SET_TEXT_EDGE(1, 0, 0, 0, 205);
	//			UI::_SET_TEXT_ENTRY("STRING");
	//			UI::_ADD_TEXT_COMPONENT_STRING(text);
	//			UI::_DRAW_TEXT(x, y);
	//			// box
	//			GRAPHICS::DRAW_RECT(x + 0.017f, y + 0.029f, 0.04f, 0.032f, 20, 20, 20, 75);
	//		}
	//	}
	//}

	//// let's add explosions to grenades in air, looks awesome !
	//DWORD mhash = 0x1152354B; // for grenade launcher use 0x741FD3C4
	//for (int i = 0; i < count; i++)
	//{
	//	Hash model = ENTITY::GET_ENTITY_MODEL(objects[i]);
	//	if (model == mhash && ENTITY::IS_ENTITY_IN_AIR(objects[i]))
	//	{
	//		Vector3 v = ENTITY::GET_ENTITY_COORDS(objects[i], TRUE);
	//		Vector3 plv = ENTITY::GET_ENTITY_COORDS(playerPed, TRUE);dio add build options

	//		// make sure that explosion won't hurt the player
	//		float dist = GAMEPLAY::GET_DISTANCE_BETWEEN_COORDS(plv.x, plv.y, plv.z, v.x, v.y, v.z, TRUE);
	//		if (dist > 10.0)
	//		{
	//			// only 1/3 of expolsions will have the sound
	//			FIRE::ADD_EXPLOSION(v.x, v.y, v.z, ExplosionTypeGrenadeL, 1.0, rand() % 3 == 0, FALSE, 0.0);
	//		}
	//	}
	//}

}


void main()
{		
	std::ofstream log_file;
	log_file.open("GTA_recording.log", std::ios_base::app);
	
	//Client client(log_file);
	//client.setUp();
	BinaryWriter binary_writer("GTA_recording.bin", log_file);

	log_file << "Each struct will have size " << sizeof(EntityState) << "." << std::endl;

	//EntityState data;

	//data.wall_time = 0.0;

	//data.id = 99;

	//data.posx = 1;
	//data.posy = 2;
	//data.posz = 3;

	//data.roll = 4;
	//data.pitch = 5;
	//data.yaw = 6;

	//data.velx = 7;
	//data.vely = 8;
	//data.velz = 9;

	//data.rvelx = 10;
	//data.rvely = 11;
	//data.rvelz = 12;
	//
	//data.collisionx = 13;
	//data.collisiony = 14;
	//data.collisionz = 15;
	//
	//data.screenx = 16;
	//data.screeny = 17;

	//data.occluded = TRUE;

	////client.sendData(data);
	//binary_writer.saveData(data);
	
	while (true)
	{
	    update(binary_writer, log_file);
		WAIT(0);
	}

	log_file.close();
	//client.cleanUp();
}


void ScriptMain()
{	
	srand(GetTickCount());
	main();
}
