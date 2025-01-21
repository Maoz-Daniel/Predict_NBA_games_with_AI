import React from 'react';
import { getTeamLogo, NBATeamName } from '../types/nba-teams';

interface TeamInfo {
  name: NBATeamName;
  record: string;
  winChance: number;
}

interface GameOfDayProps {
  homeTeam: TeamInfo;
  awayTeam: TeamInfo;
  gameTime: string;
  stadium: string;
}

export function GameOfDay({ homeTeam, awayTeam, gameTime, stadium }: GameOfDayProps) {
  const winningTeam = homeTeam.winChance > awayTeam.winChance ? homeTeam : awayTeam;
  console.log("homeTeam", homeTeam);
  console.log("awayTeam", awayTeam);
  console.log("winningTeam", winningTeam);

  return (
    <div className="bg-white p-6 rounded-lg shadow-md border-t-4 border-blue-600">
      <h3 className="text-2xl font-bold text-blue-900 mb-4 text-center">
        Game of the Day
      </h3>      
      <div className="flex justify-between items-center">
        <div className="text-center">
          <img 
            src={getTeamLogo(homeTeam.name)}
            alt={`${homeTeam.name} Logo`}
            className="w-24 h-24 mx-auto mb-2 object-contain transition-transform duration-300 hover:scale-110 filter drop-shadow-md"
          />
          <p className="font-bold text-lg">{homeTeam.name}</p>
          <p className="text-sm text-gray-600">{homeTeam.record}</p>
        </div>
        <div className="text-center mx-auto">
          <p className="text-4xl font-bold text-red-600">VS</p>
          <p className="text-sm text-gray-600 mt-2">{gameTime}</p>
          <p className="text-xs text-gray-500">{stadium}</p>
        </div>
        <div className="text-center">
          <img 
            src={getTeamLogo(awayTeam.name)}
            alt={`${awayTeam.name} Logo`}
            className="w-24 h-24 mx-auto mb-2 object-contain transition-transform duration-300 hover:scale-110 filter drop-shadow-md"
          />
          <p className="font-bold text-lg">{awayTeam.name}</p>
          <p className="text-sm text-gray-600">{awayTeam.record}</p>
        </div>
      </div>
      <div className="mt-6 bg-gradient-to-r from-blue-100 to-red-100 p-4 rounded-lg">
        <h4 className="font-bold mb-2 text-blue-900">Prediction</h4>
        <div className="flex justify-between items-center">
          <p className="text-blue-900 font-semibold">{winningTeam.name} win</p>
          <div className="w-1/2 bg-gray-200 rounded-full h-2.5">
            <div 
              className="bg-blue-600 h-2.5 rounded-full" 
              style={{ width: `${winningTeam.winChance}%` }}
            ></div>
          </div>
          <p className="text-blue-900 font-semibold">{winningTeam.winChance}%</p>
        </div>
        <p className="text-sm text-gray-700 mt-2">Based on recent performance, head-to-head history, and team statistics</p>
      </div>
    </div>
  )
}

