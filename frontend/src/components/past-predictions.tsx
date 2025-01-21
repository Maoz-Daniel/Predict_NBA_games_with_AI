import React from 'react';
import { getTeamLogo, NBATeamName } from '../types/nba-teams';

interface PastPrediction {
  id: number;
  homeTeam: NBATeamName;
  awayTeam: NBATeamName;
  prediction: NBATeamName;
  actual: NBATeamName;
}

interface PastPredictionsProps {
  predictions: PastPrediction[];
}

export function PastPredictions({ predictions }: PastPredictionsProps) {
  return (
    <div className="bg-white p-4 rounded-lg shadow-md border-t-4 border-blue-600">
      <h3 className="text-xl font-bold text-blue-900 mb-3">Past Predictions</h3>
      <ul className="space-y-3">
        {predictions.map((prediction) => (
          <li key={prediction.id} className="border-b pb-3">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center">
                <img 
                  src={getTeamLogo(prediction.homeTeam)}
                  alt={`${prediction.homeTeam} Logo`}
                  className="w-6 h-6 mr-2"
                />
                <span className="font-semibold text-sm mr-1">{prediction.homeTeam}</span>
                <span className="text-sm mx-1">vs</span>
                <span className="font-semibold text-sm ml-1">{prediction.awayTeam}</span>
                <img 
                  src={getTeamLogo(prediction.awayTeam)}
                  alt={`${prediction.awayTeam} Logo`}
                  className="w-6 h-6 ml-2"
                />
              </div>
              
            </div>
            <div className="flex justify-between items-center text-xs">
              <div>
                <p>Predicted: <span className="font-semibold">{prediction.prediction} win</span></p>
                <p>Actual: <span className="font-semibold">{prediction.actual} won</span></p>
              </div>
              <p className={`font-semibold ${(prediction.actual == prediction.prediction) ? 'text-green-600' : 'text-red-600'}`}>
                {(prediction.actual == prediction.prediction) ? 'Correct' : 'Incorrect'}
              </p>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

