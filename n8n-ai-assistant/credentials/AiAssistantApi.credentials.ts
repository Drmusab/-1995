import {
	IAuthenticateGeneric,
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class AiAssistantApi implements ICredentialType {
	name = 'aiAssistantApi';
	displayName = 'AI Assistant API';
	documentationUrl = 'https://github.com/Drmusab/-1995';
	properties: INodeProperties[] = [
		{
			displayName: 'Base URL',
			name: 'baseUrl',
			type: 'string',
			default: 'http://localhost:8000',
			placeholder: 'http://localhost:8000',
			description: 'The base URL of your AI Assistant API server',
			required: true,
		},
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: {
				password: true,
			},
			default: '',
			description: 'API key for authentication with the AI Assistant',
			required: false,
		},
		{
			displayName: 'Username',
			name: 'username',
			type: 'string',
			default: '',
			description: 'Username for basic authentication (if API key is not used)',
			required: false,
		},
		{
			displayName: 'Password',
			name: 'password',
			type: 'string',
			typeOptions: {
				password: true,
			},
			default: '',
			description: 'Password for basic authentication (if API key is not used)',
			required: false,
		},
	];

	authenticate: IAuthenticateGeneric = {
		type: 'generic',
		properties: {
			headers: {
				Authorization: '={{$credentials.apiKey ? "Bearer " + $credentials.apiKey : "Basic " + Buffer.from($credentials.username + ":" + $credentials.password).toString("base64")}}',
			},
		},
	};
}