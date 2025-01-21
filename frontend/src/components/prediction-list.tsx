import React from "react"
import { getTeamLogo, type NBATeamName } from "../types/nba-teams"

interface Prediction {
  id: number
  homeTeam: NBATeamName
  homeRecord: string
  awayTeam: NBATeamName
  awayRecord: string
  prediction: NBATeamName
  confidence: number
}

interface PredictionListProps {
  predictions: Prediction[]
}

export function PredictionList({ predictions }: PredictionListProps) {
  return (
    <div className="bg-gradient-to-br from-blue-50 to-red-50 p-6 rounded-xl shadow-lg border border-gray-200">
      <h3 className="text-3xl font-bold text-gray-800 mb-6 text-center">Today's Predictions</h3>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {predictions.map((game) => (
          <PredictionCard key={game.id} game={game} />
        ))}
      </div>
    </div>
  )
}

function PredictionCard({ game }: { game: Prediction }) {
  const isHomeTeamPredicted = game.prediction === game.homeTeam

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden transition-transform duration-300 hover:scale-105">
      <div className="p-4 bg-gradient-to-r from-blue-600 to-red-600">
        <h4 className="text-white font-semibold text-lg text-center mb-2">Game #{game.id}</h4>
      </div>
      <div className="p-4">
        <div className="flex justify-between items-center mb-4">
          <TeamInfo team={game.homeTeam} record={game.homeRecord} isHome={true} />
          <span className="text-2xl font-bold text-gray-400">VS</span>
          <TeamInfo team={game.awayTeam} record={game.awayRecord} isHome={false} />
        </div>
        <div className="mt-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-600">Prediction</span>
            <span className="text-sm font-bold text-blue-600">{game.prediction} Win</span>
          </div>
          <div className="relative pt-1">
            <div className="flex mb-2 items-center justify-between">
              <div>
                <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-blue-600 bg-blue-200">
                  {game.confidence}% Confidence
                </span>
              </div>
            </div>
            <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-gray-200">
              <div
                style={{ width: `${game.confidence}%` }}
                className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${
                  isHomeTeamPredicted ? "bg-blue-500" : "bg-red-500"
                }`}
              ></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function TeamInfo({ team, record, isHome }: { team: NBATeamName; record: string; isHome: boolean }) {
  return (
    <div className={`flex flex-col items-center ${isHome ? "order-first" : "order-last"}`}>
      <div className="relative w-16 h-16 mb-2">
                  <img 
                    src={getTeamLogo(team) || "/placeholder.svg"}
                    alt={`${team} Logo`}
                    className="w-24 h-24 mx-auto mb-2 object-contain transition-transform duration-300 hover:scale-110 filter drop-shadow-md"
                  />
      </div>
      <p className="font-semibold text-gray-800">{team}</p>
      <p className="text-xs text-gray-600">{record}</p>
    </div>
  )
}

